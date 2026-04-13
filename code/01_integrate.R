# 01_integrate.R
# Merges all stacked IPEDS components and College Scorecard into one
# longitudinal dataset, then temporally aligns inputs to outcomes.
#
# Input:  data/raw/ipeds_*.csv and data/raw/college_scorecard.csv
#         (produced by 00_stack_raw.R)
# Output: data/processed/master_aligned_raw.csv
#         data/processed/master_longitudinal.csv
#         docs/flags/*.csv  (data quality flags for review)
#
# Key data decisions (based on flag file review):
#   - Year 2024 excluded: 100% missing across all outcomes and most inputs.
#   - Sectors 7, 8, 9, 99 excluded: <2-yr institutions with near-total missingness
#     and too few observations (1-4 institutions) to contribute to analysis.
#   - Scorecard enrollment (ugds) dropped: median 25% discrepancy vs. IPEDS,
#     max 41x. IPEDS total_enrollment_ipeds is the authoritative source.
#   - Scorecard pctpell dropped after cross-check: IPEDS pell_pct is preferred.
#   - Duplicate records (GR): fixed by filtering to the four grtype codes that
#     represent total cohort / total completers (2, 4, 18, 20) and computing
#     the rate directly as completers / cohort before joining.
#   - Duplicate records (HR): fixed by filtering to the total-across-all-ranks
#     rows (siscat==1/arank==0 or siscat==100/arank==0) before joining.
#   - Duplicate records (GR200, Scorecard): caused by stacking artifacts;
#     fixed by dropping NA-unitid rows and keeping the data-rich row per key.
#   - OPEID conflicts (251 institutions): all legitimate mergers/re-brandings.
#     No action needed -- join is on unitid + year, not opeid.
#
# Author: Audrey Nichols | DATA 6950 | Spring 2026

library(tidyverse)
library(janitor)
library(naniar)

# -- Paths and settings -------------------------------------------------------
RAW_DIR  <- "data/raw"
PROC_DIR <- "data/processed"
FLAG_DIR <- "docs/flags"

dir.create(PROC_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FLAG_DIR, recursive = TRUE, showWarnings = FALSE)

# Exclude 2024 (no usable data)
YEAR_MIN <- 2015
YEAR_MAX <- 2023

# Sectors 7-9 and 99 are <2-yr and unknown -- excluded due to near-total missingness
EXCLUDE_SECTORS <- c(7, 8, 9, 99)

INCLUDE_OTHER <- FALSE   # set TRUE to keep non-4yr/2yr Carnegie institutions

# Carnegie CCBASIC codes by institution type
DOCTORAL_CODES <- c(1, 2, 3)
MASTERS_CODES  <- c(4, 5, 6)
BACC_CODES     <- c(14, 15, 16, 17)
ASSOC_CODES    <- c(18, 19, 20, 21, 22, 23)

# =============================================================================
# Helper: read a stacked CSV and clean column names
# =============================================================================
read_raw <- function(filename) {
  read_csv(file.path(RAW_DIR, filename), show_col_types = FALSE,
           na = c("", "NA", ".", "PrivacySuppressed", "PS")) |>
    clean_names() |>
    mutate(unitid = as.character(unitid))
}

# =============================================================================
# 1. Load all stacked components
# =============================================================================
message("Loading raw files...")

ipeds_hd      <- read_raw("ipeds_hd.csv")
ipeds_adm     <- read_raw("ipeds_adm.csv")
ipeds_fin_f1  <- read_raw("ipeds_fin_f1.csv")
ipeds_fin_f2  <- read_raw("ipeds_fin_f2.csv")
ipeds_fin_f3  <- read_raw("ipeds_fin_f3.csv")
ipeds_effy    <- read_raw("ipeds_effy.csv")
ipeds_ef_a    <- read_raw("ipeds_ef_a.csv")
ipeds_sfa     <- read_raw("ipeds_sfa.csv")
ipeds_gr      <- read_raw("ipeds_gr.csv") |> select(unitid, year, grtype, grtotlt)
ipeds_gr200   <- read_raw("ipeds_gr200.csv")
ipeds_hr      <- read_raw("ipeds_hr.csv")
ipeds_sal_is  <- read_raw("ipeds_sal_is.csv")
ipeds_sal_nis <- read_raw("ipeds_sal_nis.csv")
ipeds_c       <- read_raw("ipeds_c.csv")
scorecard     <- read_raw("college_scorecard.csv")

# =============================================================================
# 2. Institutional metadata from HD
# =============================================================================
# HD is the authoritative source for institution identifiers and metadata.
# Coalesce ccbasic and c21basic (2021 Carnegie update) into one column.
#
# Sector codes:
#   1 = Public 4-yr       2 = Private NP 4-yr   3 = For-profit 4-yr
#   4 = Public 2-yr       5 = Private NP 2-yr   6 = For-profit 2-yr
#   7 = Public <2-yr      8 = Private NP <2-yr  9 = For-profit <2-yr

ipeds_hd <- ipeds_hd |>
  filter(year >= YEAR_MIN, year <= YEAR_MAX) |>
  mutate(
    opeid_clean = str_pad(as.character(opeid), 8, side = "left", pad = "0"),
    sector      = as.integer(sector),
    carnegie    = coalesce(
      suppressWarnings(as.integer(c21basic)),
      suppressWarnings(as.integer(ccbasic))
    )
  ) |>
  filter(!sector %in% EXCLUDE_SECTORS) |>
  select(unitid, year, opeid_clean, instnm, sector, ccbasic, carnegie, stabbr)

message(n_distinct(ipeds_hd$unitid), " institutions retained after year and sector filters")

# Flag institutions with multiple OPEIDs across years (mergers / re-brandings).
# These are expected and do not affect the analysis since we join on unitid + year.
opeid_conflicts <- ipeds_hd |>
  group_by(unitid) |>
  summarise(n_opeids = n_distinct(opeid_clean), .groups = "drop") |>
  filter(n_opeids > 1)

if (nrow(opeid_conflicts) > 0) {
  message(nrow(opeid_conflicts), " institutions have conflicting OPEIDs (expected -- mergers/re-brandings)")
  write_csv(opeid_conflicts, file.path(FLAG_DIR, "opeid_conflicts.csv"))
}

# =============================================================================
# 3. Finance -- harmonize GASB and FASB forms to common column names
# =============================================================================
# Three forms cover different institution types:
#   F1 = GASB (public institutions)
#   F2 = FASB (private nonprofit)
#   F3 = FASB (for-profit)
#
# Column names after clean_names():
#   F1: f1c011 = instruction, f1c061 = student services, f1n07 = total exp
#   F2: f2e011 = instruction, f2e051 = student services, f2b02 = total exp
#   F3: f3e011 = instruction, f3e03b1 = student services, f3b02 = total exp

extract_fin <- function(df, instr_col, stud_col, total_col, form_label) {
  df |>
    transmute(
      unitid,
      year,
      instr_exp     = as.numeric({{ instr_col }}),
      stud_serv_exp = as.numeric({{ stud_col }}),
      total_exp     = as.numeric({{ total_col }}),
      finance_form  = form_label
    )
}

fin_combined <- bind_rows(
  extract_fin(ipeds_fin_f1, f1c011, f1c061, f1n07,    "GASB"),
  extract_fin(ipeds_fin_f2, f2e011, f2e051, f2b02,    "FASB_NP"),
  extract_fin(ipeds_fin_f3, f3e011, f3e03b1, f3b02,   "FASB_FP")
)

fin_overlaps <- fin_combined |>
  group_by(unitid, year) |>
  filter(n() > 1) |>
  ungroup()

if (nrow(fin_overlaps) > 0) {
  warning(nrow(fin_overlaps), " institution-year records appear in multiple finance forms")
  write_csv(fin_overlaps, file.path(FLAG_DIR, "finance_form_overlaps.csv"))
  # Keep only one row per institution-year: prefer GASB, then FASB_NP, then FASB_FP
  fin_combined <- fin_combined |>
    mutate(form_order = case_when(
      finance_form == "GASB"    ~ 1L,
      finance_form == "FASB_NP" ~ 2L,
      finance_form == "FASB_FP" ~ 3L,
      TRUE                      ~ 4L
    )) |>
    group_by(unitid, year) |>
    slice_min(form_order, n = 1, with_ties = FALSE) |>
    ungroup() |>
    select(-form_order)
  message("Finance deduplication: kept one row per institution-year")
}

rm(ipeds_fin_f1, ipeds_fin_f2, ipeds_fin_f3)

# =============================================================================
# 4. Prepare individual IPEDS components
# =============================================================================

# 12-month FTE enrollment (EFFY)
# effylev 1 = total, 2 = undergraduate, 4 = graduate
ipeds_fte <- ipeds_effy |>
  mutate(effylev = as.integer(effylev), efytotlt = as.numeric(efytotlt)) |>
  filter(effylev %in% c(1, 2, 4)) |>
  select(unitid, year, effylev, efytotlt) |>
  pivot_wider(names_from = effylev, values_from = efytotlt, names_prefix = "fte_level") |>
  rename(total_fte = fte_level1, fte_ug = fte_level2, fte_grad = fte_level4)

rm(ipeds_effy)

# Fall enrollment headcount (EF_A)
# efalevel == 1 is the "all students" total row
ipeds_headcount <- ipeds_ef_a |>
  mutate(efalevel = as.integer(efalevel), eftotlt = as.numeric(eftotlt)) |>
  filter(efalevel == 1) |>
  select(unitid, year, total_enrollment_ipeds = eftotlt)

rm(ipeds_ef_a)

# Pell grant share (SFA)
# pgrnt_p = % of undergrads receiving Pell grants (stored as 0-100 in IPEDS)
ipeds_pell <- ipeds_sfa |>
  transmute(unitid, year, pell_pct = as.numeric(pgrnt_p))

rm(ipeds_sfa)

# Admissions (ADM)
# admit_rate  = admitted (admssn) / applied (applcn)
# sat75       = mean of SAT verbal + math 75th percentile scores
# act75       = ACT composite 75th percentile
# yield_rate  = enrolled (enrlt) / admitted (admssn)
# Only 4-year institutions submit the ADM survey; 2-year institutions are
# typically open-admission and will have NA for all four columns -- expected.
ipeds_admissions <- ipeds_adm |>
  transmute(
    unitid,
    year,
    applcn = as.numeric(applcn),
    admssn = as.numeric(admssn),
    enrlt  = as.numeric(enrlt),
    satvr75 = as.numeric(satvr75),
    satmt75 = as.numeric(satmt75),
    actcm75 = as.numeric(actcm75)
  ) |>
  mutate(
    admit_rate = if_else(applcn > 0, admssn / applcn, NA_real_),
    sat75      = if_else(!is.na(satvr75) | !is.na(satmt75),
                         rowMeans(cbind(satvr75, satmt75), na.rm = TRUE),
                         NA_real_),
    act75      = actcm75,
    yield_rate = if_else(admssn > 0, enrlt / admssn, NA_real_)
  ) |>
  select(unitid, year, admit_rate, sat75, act75, yield_rate)

rm(ipeds_adm)

# 150% graduation rates (GR)
# GRTYPE codes (verified against IPEDS data dictionary and raw row counts):
#   grtype 2  = bachelor's-seeking revised cohort (4-year institutions)
#   grtype 3  = ALL bachelor's-seeking completers within 150% of normal time
#               (the standard 6-year graduation rate numerator)
#   grtype 4  = bachelor's-seeking completers within 150% EXCLUDING transfer-outs
#               (a subset of grtype 3 -- DO NOT use for graduation rate)
#   grtype 18 = associate's-seeking revised cohort (2-year institutions)
#   grtype 20 = associate's-seeking completers within 150% (3-yr rate numerator)
# grad_rate_150_4yr = grtype_3 / grtype_2  (all completers / cohort)
# grad_rate_150_2yr = grtype_20 / grtype_18

ipeds_gr150 <- ipeds_gr |>
  mutate(grtype  = as.integer(grtype),
         grtotlt = as.numeric(grtotlt)) |>
  filter(grtype %in% c(2L, 3L, 18L, 20L)) |>
  group_by(unitid, year, grtype) |>
  summarise(grtotlt = mean(grtotlt, na.rm = TRUE), .groups = "drop") |>
  pivot_wider(names_from = grtype, values_from = grtotlt, names_prefix = "grtype_")

# Defensively add columns if a grtype was absent in this dataset
for (.v in c("grtype_2", "grtype_3", "grtype_18", "grtype_20")) {
  if (!.v %in% names(ipeds_gr150)) ipeds_gr150[[.v]] <- NA_real_
}

ipeds_gr150 <- ipeds_gr150 |>
  transmute(
    unitid,
    year,
    grad_rate_150_4yr = if_else(
      !is.na(grtype_2) & grtype_2 > 0, grtype_3 / grtype_2, NA_real_
    ),
    grad_rate_150_2yr = if_else(
      !is.na(grtype_18) & grtype_18 > 0, grtype_20 / grtype_18, NA_real_
    )
  )

message("GR150 prepared: ", nrow(ipeds_gr150), " institution-year rows")
# grad_rate_150_4yr = 6-yr rate for 4-year institutions (0-1 proportion)
# grad_rate_150_2yr = 3-yr rate for 2-year institutions (0-1 proportion)

rm(ipeds_gr)

# 200% graduation rates (GR200)
# bagr200 = 8-yr rate for 4-year institutions; l4gr200 = 200% rate for 2-year
# The stacked GR200 file contains a duplicate UNITID column (unitid_2) from the
# stacking step, which produces 4,954 rows with unitid=NA. Filter these out
# before use. Also deduplicate in case stacking introduced any other extras.
ipeds_gr200_clean <- ipeds_gr200 |>
  filter(!is.na(unitid)) |>
  group_by(unitid, year) |>
  slice_head(n = 1) |>
  ungroup() |>
  transmute(unitid, year,
            gr200_4yr = as.numeric(bagr200),
            gr200_2yr = as.numeric(l4gr200))

rm(ipeds_gr200)

# Faculty FTE (HR)
# The HR file has one row per (unitid, year, siscat, facstat, arank) --
# 65 combinations per institution-year. Two combinations represent the total
# instructional staff count across all ranks:
#   siscat == 1,   facstat == 0,  arank == 0  (legacy survey years)
#   siscat == 100, facstat == 10, arank == 0  (newer survey years)
# These are mutually exclusive across years; slice_max is a safety net only.
ipeds_faculty <- ipeds_hr |>
  filter(
    (siscat == 1   & facstat == 0  & arank == 0) |
    (siscat == 100 & facstat == 10 & arank == 0)
  ) |>
  group_by(unitid, year) |>
  slice_max(as.numeric(hrtotlt), n = 1, with_ties = FALSE) |>
  ungroup() |>
  transmute(unitid, year, faculty_fte = as.numeric(hrtotlt))

rm(ipeds_hr)

# Instructional staff salaries (SAL_IS)
# arank == 7 is the total-across-all-ranks row
# saeq9at = average salary equated to 9-month contract (best for cross-institution comparison)
ipeds_sal_instr <- ipeds_sal_is |>
  mutate(arank = as.integer(arank)) |>
  filter(arank == 7L) |>
  transmute(unitid, year,
            sal_instr_outlay  = as.numeric(saoutlt),
            sal_instr_nstaff  = as.numeric(satotlt),
            sal_instr_avg_9mo = as.numeric(saeq9at))

rm(ipeds_sal_is)

# Non-instructional staff salaries (SAL_NIS)
# sanin01 = headcount; sanit01 = total salary outlays
ipeds_sal_noninstr <- ipeds_sal_nis |>
  transmute(unitid, year,
            sal_noninstr_nstaff = as.numeric(sanin01),
            sal_noninstr_outlay = as.numeric(sanit01)) |>
  mutate(sal_noninstr_avg = if_else(sal_noninstr_nstaff > 0,
                                    sal_noninstr_outlay / sal_noninstr_nstaff,
                                    NA_real_))

rm(ipeds_sal_nis)

# Total completions (C_A)
# Sum ctotalt across all CIP codes and award levels per institution-year
ipeds_completions <- ipeds_c |>
  mutate(ctotalt = as.numeric(ctotalt)) |>
  group_by(unitid, year) |>
  summarise(total_completions = sum(ctotalt, na.rm = TRUE), .groups = "drop")

rm(ipeds_c)

# =============================================================================
# 5. Merge all IPEDS components
# =============================================================================
# HD is the spine -- every Title IV institution appears here each year.
# All joins are left_join so institutions with missing component data are
# retained with NAs rather than silently dropped.

message("Merging IPEDS components...")

ipeds_merged <- ipeds_hd |>
  left_join(fin_combined,        by = c("unitid", "year")) |>
  left_join(ipeds_fte,           by = c("unitid", "year")) |>
  left_join(ipeds_headcount,     by = c("unitid", "year")) |>
  left_join(ipeds_pell,          by = c("unitid", "year")) |>
  left_join(ipeds_gr150,         by = c("unitid", "year")) |>
  left_join(ipeds_gr200_clean,   by = c("unitid", "year")) |>
  left_join(ipeds_faculty,       by = c("unitid", "year")) |>
  left_join(ipeds_sal_instr,     by = c("unitid", "year")) |>
  left_join(ipeds_sal_noninstr,  by = c("unitid", "year")) |>
  left_join(ipeds_completions,   by = c("unitid", "year")) |>
  left_join(ipeds_admissions,   by = c("unitid", "year")) |>
  mutate(
    instr_exp_per_fte     = if_else(total_fte > 0, instr_exp     / total_fte, NA_real_),
    stud_serv_exp_per_fte = if_else(total_fte > 0, stud_serv_exp / total_fte, NA_real_),
    stud_fac_ratio        = if_else(faculty_fte > 0, total_fte / faculty_fte, NA_real_),
    total_sal_per_fte     = if_else(
      total_fte > 0,
      (coalesce(sal_instr_outlay, 0) + coalesce(sal_noninstr_outlay, 0)) / total_fte,
      NA_real_
    )
  )

rm(fin_combined, ipeds_fte, ipeds_headcount, ipeds_pell, ipeds_gr150,
   ipeds_gr200_clean, ipeds_faculty, ipeds_sal_instr, ipeds_sal_noninstr,
   ipeds_completions, ipeds_admissions)
gc()

# Verify the GR fix worked -- should be zero duplicates
n_dupes <- ipeds_merged |> group_by(unitid, year) |> filter(n() > 1) |> nrow()
if (n_dupes > 0) {
  warning(n_dupes, " duplicate unitid-year rows remain -- check GR file processing")
} else {
  message("No duplicate unitid-year rows in merged IPEDS dataset")
}

# =============================================================================
# 6. Prepare College Scorecard
# =============================================================================
# Scorecard enrollment (ugds) is NOT carried forward -- it showed a median 25%
# discrepancy vs. IPEDS headcount and a max of 41x difference.
# Scorecard pctpell is kept temporarily for cross-check only, then dropped.

scorecard_clean <- scorecard |>
  mutate(
    across(c(md_earn_wne_p6, md_earn_wne_p10, rpy_3yr_rt,
             c150_4, c150_l4, pctpell,
             ret_ft4_pooled, ret_ftl4_pooled),
           as.numeric),
    completion_rate_sc = coalesce(c150_4, c150_l4),
    retention_rate_sc  = coalesce(ret_ft4_pooled, ret_ftl4_pooled)
  ) |>
  select(unitid, year,
         median_earnings_6yr  = md_earn_wne_p6,
         median_earnings_10yr = md_earn_wne_p10,
         loan_repayment_3yr   = rpy_3yr_rt,
         completion_rate_sc,
         retention_rate_sc,
         pctpell_sc           = pctpell)

rm(scorecard)

# Deduplicate Scorecard to one row per unitid + year before joining.
# 6,429 pairs (all in 2023) have exactly two rows: one with actual data and
# one with near-total NAs (a stacking artifact from the raw Scorecard files).
# slice(1) is non-deterministic; instead prefer the row with the fewest NAs.
scorecard_clean <- scorecard_clean |>
  filter(!is.na(unitid)) |>
  mutate(.na_count = rowSums(is.na(across(everything())))) |>
  group_by(unitid, year) |>
  slice_min(.na_count, n = 1, with_ties = FALSE) |>
  ungroup() |>
  select(-.na_count)

message("Scorecard prepared: ", nrow(scorecard_clean), " institution-year rows")

# =============================================================================
# 7. Join IPEDS and Scorecard
# =============================================================================
message("Joining IPEDS and Scorecard...")

master <- ipeds_merged |>
  left_join(scorecard_clean, by = c("unitid", "year"))

rm(ipeds_merged, scorecard_clean)
gc()

# Cross-check Pell grant share then drop Scorecard version
# IPEDS reports 0-100; Scorecard reports 0-1 proportion
pell_discrepancies <- master |>
  filter(!is.na(pell_pct), !is.na(pctpell_sc)) |>
  mutate(pell_diff = abs(pell_pct / 100 - pctpell_sc)) |>
  filter(pell_diff > 0.05) |>
  select(unitid, instnm, year, pell_pct, pctpell_sc, pell_diff) |>
  arrange(desc(pell_diff))

message(nrow(pell_discrepancies), " institution-years have >5% Pell discrepancy between IPEDS and Scorecard")
write_csv(pell_discrepancies, file.path(FLAG_DIR, "pell_discrepancies.csv"))

master <- master |> select(-pctpell_sc)

message("Master dataset: ", n_distinct(master$unitid), " institutions, ",
        n_distinct(master$year), " years, ", nrow(master), " rows total")

# =============================================================================
# 8. Degree level classification and temporal alignment
# =============================================================================
# Match inputs at year t to outcomes at year t + lag.
#   4-year institutions (Doctoral, Master's, Baccalaureate): lag = 4
#   2-year institutions (Associate's):                       lag = 2

master <- master |>
  mutate(
    ccbasic      = as.integer(ccbasic),
    degree_level = case_when(
      ccbasic %in% c(DOCTORAL_CODES, MASTERS_CODES, BACC_CODES) ~ "4-year",
      ccbasic %in% ASSOC_CODES                                   ~ "2-year",
      TRUE                                                        ~ "other"
    )
  )

# Flag genuine degree level changes (excludes the 2024 Carnegie coding artifact,
# which is already removed by the year filter above)
level_changes <- master |>
  group_by(unitid) |>
  arrange(year) |>
  mutate(base_level = first(degree_level)) |>
  filter(degree_level != base_level) |>
  select(unitid, instnm, year, base_level, degree_level) |>
  ungroup()

if (nrow(level_changes) > 0) {
  message(n_distinct(level_changes$unitid), " institutions changed degree level -- see flags/degree_level_changes.csv")
  write_csv(level_changes, file.path(FLAG_DIR, "degree_level_changes.csv"))
}

outcome_lags <- master |>
  group_by(unitid) |>
  arrange(year) |>
  summarise(
    base_level  = first(degree_level),
    outcome_lag = if_else(first(degree_level) == "4-year", 4L, 2L),
    .groups     = "drop"
  )

master <- master |> left_join(outcome_lags, by = "unitid")

inputs <- master |>
  mutate(outcome_year = year + outcome_lag) |>
  select(unitid, input_year = year, outcome_year, outcome_lag,
         instnm, sector, ccbasic, stabbr, base_level, finance_form,
         instr_exp_per_fte, stud_serv_exp_per_fte, total_exp,
         total_fte, total_enrollment_ipeds, faculty_fte, stud_fac_ratio,
         pell_pct, sal_instr_avg_9mo, sal_noninstr_avg, total_sal_per_fte,
         total_completions, admit_rate, sat75, act75, yield_rate)

outcomes <- master |>
  select(unitid, outcome_year = year,
         grad_rate_150_4yr,                 # 0-1 proportion, computed from GR counts
         grad_rate_150_2yr,                 # 0-1 proportion
         grad_rate_200_4yr = gr200_4yr,
         grad_rate_200_2yr = gr200_2yr,
         median_earnings_6yr, median_earnings_10yr,
         loan_repayment_3yr, completion_rate_sc, retention_rate_sc)

aligned <- inputs |>
  left_join(outcomes, by = c("unitid", "outcome_year"))

if (!INCLUDE_OTHER) aligned <- aligned |> filter(base_level != "other")

message("Aligned dataset: ", nrow(aligned), " input-outcome pairs, ",
        n_distinct(aligned$unitid), " institutions")

# =============================================================================
# 9. Data quality assessment
# =============================================================================
message("Running data quality checks...")

miss_summary <- aligned |>
  miss_var_summary() |>
  arrange(desc(pct_miss))

print(miss_summary, n = 40)
write_csv(miss_summary, file.path(FLAG_DIR, "missingness_summary.csv"))

miss_by_year <- aligned |>
  group_by(input_year) |>
  summarise(
    across(
      c(median_earnings_6yr, loan_repayment_3yr,
        grad_rate_150_4yr, grad_rate_200_4yr,
        instr_exp_per_fte, stud_serv_exp_per_fte),
      ~ round(mean(is.na(.)) * 100, 1),
      .names = "pct_miss_{.col}"
    ),
    n_institutions = n_distinct(unitid),
    .groups = "drop"
  )

print(miss_by_year)
write_csv(miss_by_year, file.path(FLAG_DIR, "missingness_by_year.csv"))

miss_by_sector <- aligned |>
  group_by(sector) |>
  summarise(
    across(
      c(median_earnings_6yr, loan_repayment_3yr,
        grad_rate_150_4yr, instr_exp_per_fte),
      ~ round(mean(is.na(.)) * 100, 1),
      .names = "pct_miss_{.col}"
    ),
    n_institutions = n_distinct(unitid),
    .groups = "drop"
  )

print(miss_by_sector)
write_csv(miss_by_sector, file.path(FLAG_DIR, "missingness_by_sector.csv"))

# Duplicate check -- should be zero now
duplicates <- aligned |>
  group_by(unitid, input_year) |>
  filter(n() > 1) |>
  ungroup()

if (nrow(duplicates) > 0) {
  warning(nrow(duplicates), " duplicate unitid-year records remain")
  write_csv(duplicates, file.path(FLAG_DIR, "duplicate_records.csv"))
} else {
  message("No duplicate unitid-year records")
}

# Outlier detection
# stud_serv uses 99th percentile threshold (tighter) -- Strayer showed this
# variable is more prone to reporting anomalies than other expenditure vars
p99_stud_serv <- quantile(aligned$stud_serv_exp_per_fte, 0.99, na.rm = TRUE)
p99_x3        <- function(x) x > 3 * quantile(x, 0.99, na.rm = TRUE)

outlier_records <- aligned |>
  mutate(
    flag_instr_exp    = p99_x3(instr_exp_per_fte),
    flag_stud_serv    = stud_serv_exp_per_fte > p99_stud_serv,
    flag_sal_instr    = p99_x3(sal_instr_avg_9mo),
    flag_sal_noninstr = p99_x3(sal_noninstr_avg)
  ) |>
  filter(flag_instr_exp | flag_stud_serv | flag_sal_instr | flag_sal_noninstr) |>
  select(unitid, instnm, input_year,
         instr_exp_per_fte, stud_serv_exp_per_fte,
         sal_instr_avg_9mo, sal_noninstr_avg,
         starts_with("flag_")) |>
  arrange(desc(stud_serv_exp_per_fte))

message(nrow(outlier_records), " institution-years flagged as expenditure/salary outliers")
write_csv(outlier_records, file.path(FLAG_DIR, "expenditure_outliers.csv"))

# =============================================================================
# 10. Save outputs
# =============================================================================
write_csv(aligned, file.path(PROC_DIR, "master_aligned_raw.csv"))
write_csv(master,  file.path(PROC_DIR, "master_longitudinal.csv"))

message("=================================================================")
message("Integration complete.")
message("  master_aligned_raw.csv  -- ", nrow(aligned), " rows (", n_distinct(aligned$unitid), " institutions)")
message("  master_longitudinal.csv -- ", nrow(master), " rows (full panel)")
message("  Review docs/flags/ before running 02_clean.R")
message("=================================================================")
