# 00_stack_raw.R
# Stacks annual IPEDS CSV files and College Scorecard files into one CSV per
# component. Run this once before the cleaning/integration scripts.
#
# Expected folder structure:
#   data/raw/ipeds/        -- unzipped IPEDS CSVs (e.g. hd2019.csv, f1920_f1a.csv)
#   data/raw/scorecard/    -- Scorecard files (e.g. MERGED2019_20_PP.csv)
#   data/raw/              -- output goes here
#
# Author: Audrey Nichols | DATA 6950 | Spring 2026

library(tidyverse)

# -- Paths --------------------------------------------------------------------
IPEDS_DIR <- "data/raw/ipeds"
SC_DIR    <- "data/raw/college scorecard"
OUT_DIR   <- "data/raw"

YEARS <- 2015:2024

# =============================================================================
# Stacking function
# =============================================================================
# Reads one CSV per year, adds a 'year' column, and binds them together.
# All columns are read as character to avoid type conflicts across years.
#
# name_format controls how the filename is constructed from the year:
#   "simple"     -- {prefix}{YYYY}{suffix}.csv      e.g. hd2019.csv
#   "span"       -- {prefix}{YY}{YY+1}{suffix}.csv  e.g. f1920_f1a.csv
#   "short_year" -- {prefix}{YY}{suffix}.csv         e.g. gr200_19.csv
#   "sfa"        -- sfa{YY}{YY+1}.csv                e.g. sfa1920.csv

build_filename <- function(prefix, suffix, yr, name_format) {
  yy      <- str_sub(as.character(yr), 3, 4)
  yy_next <- str_pad(as.integer(yy) + 1, 2, pad = "0")

  switch(name_format,
    simple     = paste0(prefix, yr, suffix, ".csv"),
    span       = paste0(prefix, yy, yy_next, suffix, ".csv"),
    short_year = paste0(prefix, yy, suffix, ".csv"),
    sfa        = paste0(prefix, yy, yy_next, ".csv")
  )
}

stack_component <- function(prefix, output_name, years = YEARS,
                            suffix = "", name_format = "simple") {
  rows <- map(years, function(yr) {
    fname <- build_filename(prefix, suffix, yr, name_format)
    fpath <- file.path(IPEDS_DIR, fname)

    if (!file.exists(fpath)) {
      warning("Missing: ", fpath, " -- skipping year ", yr)
      return(NULL)
    }

    read_csv(fpath, show_col_types = FALSE, col_types = cols(.default = "c")) |>
      mutate(year = yr)
  })

  out <- bind_rows(rows)

  if (nrow(out) == 0) {
    warning("No files found for ", output_name, " -- check paths and naming pattern")
    return(invisible(NULL))
  }

  write_csv(out, file.path(OUT_DIR, output_name))
  message("Saved ", nrow(out), " rows -> ", output_name)
  invisible(out)
}

# =============================================================================
# Stack IPEDS components
# =============================================================================
message("--- Stacking IPEDS ---")

# Institutional characteristics and enrollment
stack_component("hd",   "ipeds_hd.csv")
stack_component("adm",  "ipeds_adm.csv")
stack_component("ic",   "ipeds_ic.csv")
stack_component("ef",   "ipeds_ef_a.csv",    suffix = "a")
stack_component("effy", "ipeds_effy.csv")

# Finance -- three form types covering different institution categories
stack_component("f", "ipeds_fin_f1.csv", suffix = "_f1a", name_format = "span")
stack_component("f", "ipeds_fin_f2.csv", suffix = "_f2",  name_format = "span")
stack_component("f", "ipeds_fin_f3.csv", suffix = "_f3",  name_format = "span")

# Completions, graduation rates, and student financial aid
stack_component("c",      "ipeds_c.csv",      suffix = "_a")
stack_component("gr",     "ipeds_gr.csv")
stack_component("gr200_", "ipeds_gr200.csv",  name_format = "short_year")
stack_component("sfa",    "ipeds_sfa.csv",    name_format = "sfa")

# Human resources and salaries
stack_component("s",   "ipeds_hr.csv",      suffix = "_is")
stack_component("sal", "ipeds_sal_is.csv",  suffix = "_is")
stack_component("sal", "ipeds_sal_nis.csv", suffix = "_nis")

# =============================================================================
# Stack College Scorecard
# =============================================================================
message("--- Stacking College Scorecard ---")

sc_files <- list.files(SC_DIR, pattern = "MERGED.*\\.csv$",
                       full.names = TRUE, ignore.case = TRUE)

if (length(sc_files) == 0) {
  warning("No Scorecard files found in ", SC_DIR)
} else {
  sc_out <- map(sc_files, function(f) {
    yr <- as.integer(str_extract(basename(f), "\\d{4}"))
    read_csv(f, show_col_types = FALSE,
             col_types = cols(.default = "c"),
             na = c("", "NA", "NULL", "PrivacySuppressed")) |>
      mutate(year = yr)
  }) |> bind_rows()

  write_csv(sc_out, file.path(OUT_DIR, "college_scorecard.csv"))
  message("Saved ", nrow(sc_out), " rows -> college_scorecard.csv")
}

# =============================================================================
# Row count check
# =============================================================================
message("--- Row counts ---")

expected <- c(
  "ipeds_hd.csv", "ipeds_adm.csv", "ipeds_ic.csv", "ipeds_ef_a.csv", "ipeds_effy.csv",
  "ipeds_fin_f1.csv", "ipeds_fin_f2.csv", "ipeds_fin_f3.csv",
  "ipeds_c.csv", "ipeds_gr.csv", "ipeds_gr200.csv", "ipeds_sfa.csv",
  "ipeds_hr.csv", "ipeds_sal_is.csv", "ipeds_sal_nis.csv",
  "college_scorecard.csv"
)

walk(expected, function(f) {
  path <- file.path(OUT_DIR, f)
  if (file.exists(path)) {
    n <- nrow(read_csv(path, show_col_types = FALSE, col_types = cols(.default = "c")))
    message("  ", f, ": ", format(n, big.mark = ","), " rows")
  } else {
    warning("  MISSING: ", f)
  }
})

message("--- Done ---")
