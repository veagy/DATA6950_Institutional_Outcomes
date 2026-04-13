# 03_efficiency.R
# Runs DEA (CCR and BCC) and SFA (Cobb-Douglas) efficiency analysis on the
# cleaned dataset, classifies institutions into efficiency tiers, and exports
# scores, tier profiles, and diagnostic plots.
#
# Input:  data/processed/analysis_ready.csv  (from 02_clean.R)
# Output: data/processed/efficiency_scores.csv
#         output/efficiency/tier_profiles.csv
#         output/efficiency/efficiency_plots.pdf
#         (sensitivity versions of the above with _no_forprofit suffix)
#
# DEA inputs:  instr_exp_per_fte, stud_serv_exp_per_fte, stud_fac_ratio, pell_pct
# DEA outputs: grad_rate_150_4yr (required), median_earnings_6yr, loan_repayment_3yr
#              (optional -- included if >= 100 non-missing observations)
#
# Author: Audrey Nichols | DATA 6950 | Spring 2026

library(tidyverse)
library(Benchmarking)
library(frontier)
library(gridExtra)

# -- Paths and settings -------------------------------------------------------
PROC_DIR   <- "data/processed"
OUTPUT_DIR <- "output/efficiency"
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# For-profit sector codes (IPEDS)
FORPROFIT_CODES <- c(3, 6, 9)

SECTOR_LABELS <- c(
  "1" = "Public 4-yr",      "2" = "Private NP 4-yr", "3" = "For-profit 4-yr",
  "4" = "Public 2-yr",      "5" = "Private NP 2-yr", "6" = "For-profit 2-yr"
)

# DEA input variables -- must all be strictly positive
DEA_INPUTS <- c("instr_exp_per_fte", "stud_serv_exp_per_fte",
                "stud_fac_ratio", "pell_pct")

# =============================================================================
# 1. Load and prepare
# =============================================================================
# DEA requires one row per institution, so we use the most recent input year
# with a valid graduation rate for each institution.

message("Loading data...")
raw <- read_csv(file.path(PROC_DIR, "analysis_ready.csv"), show_col_types = FALSE)
message(nrow(raw), " rows, ", n_distinct(raw$unitid), " institutions")

# Collapse to one row per institution: most recent year with graduation rate data
df <- raw |>
  filter(!is.na(grad_rate_150_4yr)) |>
  group_by(unitid) |>
  slice_max(input_year, n = 1, with_ties = FALSE) |>
  ungroup()

message(nrow(df), " institutions with graduation rate data")

# Rescale pell_pct if still on 0-100 scale
if (max(df$pell_pct, na.rm = TRUE) > 1) {
  df$pell_pct <- df$pell_pct / 100
}

# =============================================================================
# 2. Build analysis subset
# =============================================================================
# Determines which DEA outputs to use based on data availability, filters to
# complete cases on all required variables, and enforces the DEA requirement
# that all inputs and outputs are strictly positive.

build_subset <- function(df, exclude_forprofit = FALSE) {

  # Add optional outputs if enough data exists
  outputs_use <- "grad_rate_150_4yr"
  for (col in c("median_earnings_6yr", "loan_repayment_3yr")) {
    if (col %in% names(df) && sum(!is.na(df[[col]])) >= 100) {
      outputs_use <- c(outputs_use, col)
    }
  }
  message("DEA outputs: ", paste(outputs_use, collapse = ", "))

  required <- unique(c(DEA_INPUTS, outputs_use))
  sub <- df |> filter(if_all(all_of(intersect(required, names(df))), ~ !is.na(.)))

  if (nrow(sub) < 30) stop("Only ", nrow(sub), " complete cases -- too few for DEA")

  if (exclude_forprofit) {
    n_before <- nrow(sub)
    sub <- sub |> filter(!sector %in% FORPROFIT_CODES)
    message("Removed ", n_before - nrow(sub), " for-profit institutions")
  }

  # All DEA variables must be strictly positive
  for (col in intersect(required, names(sub))) {
    sub[[col]] <- pmax(as.numeric(sub[[col]]), 1e-6, na.rm = TRUE)
  }

  message(nrow(sub), " institutions in DEA/SFA subset")
  attr(sub, "outputs_use") <- outputs_use
  sub
}

# =============================================================================
# 3. DEA
# =============================================================================
run_dea <- function(df) {
  outputs_use <- attr(df, "outputs_use") %||% "grad_rate_150_4yr"

  X <- as.matrix(df[DEA_INPUTS])       |> apply(2, as.numeric)
  Y <- as.matrix(df[outputs_use])      |> apply(2, as.numeric)

  message("DEA: ", nrow(df), " institutions | ", ncol(X), " inputs | ", ncol(Y), " outputs")

  df$dea_ccr_score <- tryCatch(
    eff(dea(X, Y, RTS = "crs", ORIENTATION = "in")),
    error = function(e) { message("DEA CCR failed: ", e$message); NA_real_ }
  )
  df$dea_bcc_score <- tryCatch(
    eff(dea(X, Y, RTS = "vrs", ORIENTATION = "in")),
    error = function(e) { message("DEA BCC failed: ", e$message); NA_real_ }
  )

  message(sprintf("DEA done  CCR mean=%.4f  BCC mean=%.4f",
                  mean(df$dea_ccr_score, na.rm = TRUE),
                  mean(df$dea_bcc_score, na.rm = TRUE)))

  # Sector summary
  if ("sector" %in% names(df)) {
    df |>
      mutate(sec = recode(as.character(sector), !!!SECTOR_LABELS, .default = as.character(sector))) |>
      group_by(sec) |>
      summarise(n   = n(),
                ccr = round(mean(dea_ccr_score, na.rm = TRUE), 3),
                bcc = round(mean(dea_bcc_score, na.rm = TRUE), 3),
                .groups = "drop") |>
      print()
  }

  df
}

# =============================================================================
# 4. SFA (Cobb-Douglas)
# =============================================================================
# Translog is also attempted as a robustness check but often fails to converge
# on this dataset, so Cobb-Douglas is the primary specification.

run_sfa <- function(df) {
  fmla <- as.formula(paste(
    "log(grad_rate_150_4yr) ~",
    paste(paste0("log(", DEA_INPUTS, ")"), collapse = " + ")
  ))

  message("SFA: estimating Cobb-Douglas frontier...")

  result <- tryCatch({
    fit    <- sfa(fmla, data = df, ineffDecrease = FALSE)
    scores <- as.numeric(efficiencies(fit))
    list(scores = scores, fit = fit)
  }, error = function(e) {
    message("SFA failed: ", e$message)
    NULL
  })

  if (!is.null(result) && length(result$scores) == nrow(df)) {
    df$sfa_cd_score <- result$scores
    message(sprintf("SFA done  mean=%.4f  sd=%.4f",
                    mean(df$sfa_cd_score, na.rm = TRUE),
                    sd(df$sfa_cd_score, na.rm = TRUE)))
  } else {
    df$sfa_cd_score <- NA_real_
    message("SFA scores unavailable -- continuing with DEA only")
  }

  df
}

# =============================================================================
# 5. Tier classification
# =============================================================================
# Institutions are split into tertiles (High / Mid / Low) using the average of
# all available efficiency scores. Spearman correlation between DEA-BCC and
# SFA-CD is reported as the primary cross-method agreement measure.

classify_tiers <- function(df) {
  avail_scores <- intersect(c("dea_bcc_score", "dea_ccr_score", "sfa_cd_score"), names(df))
  avail_scores <- avail_scores[sapply(avail_scores, function(x) sum(!is.na(df[[x]])) > 10)]

  if (length(avail_scores) == 0) {
    message("No efficiency scores available for tier classification")
    return(df)
  }

  df$avg_efficiency_score <- rowMeans(df[avail_scores], na.rm = TRUE)

  q <- quantile(df$avg_efficiency_score, probs = c(1/3, 2/3), na.rm = TRUE)
  df$efficiency_tier <- case_when(
    df$avg_efficiency_score >= q[2] ~ "High-Efficiency",
    df$avg_efficiency_score >= q[1] ~ "Mid-Efficiency",
    TRUE                            ~ "Low-Efficiency"
  ) |> factor(levels = c("High-Efficiency", "Mid-Efficiency", "Low-Efficiency"))

  message("Tier distribution:")
  print(table(df$efficiency_tier))

  # DEA-BCC vs SFA-CD agreement
  if (all(c("dea_bcc_score", "sfa_cd_score") %in% names(df))) {
    valid <- df |> filter(!is.na(dea_bcc_score), !is.na(sfa_cd_score))
    if (nrow(valid) >= 10) {
      rho <- cor(valid$dea_bcc_score, valid$sfa_cd_score, method = "spearman")
      message(sprintf("Spearman rho (DEA-BCC vs SFA-CD): %.4f", rho))
    }
  }

  df
}

# =============================================================================
# 6. Export results
# =============================================================================
export_results <- function(df, suffix = "full") {
  score_cols <- intersect(
    c("unitid", "instnm", "stabbr", "sector", "input_year",
      DEA_INPUTS, "grad_rate_150_4yr",
      "dea_ccr_score", "dea_bcc_score", "sfa_cd_score",
      "avg_efficiency_score", "efficiency_tier"),
    names(df)
  )

  write_csv(df[score_cols],
            file.path(PROC_DIR, paste0("efficiency_scores_", suffix, ".csv")))
  message("Saved efficiency_scores_", suffix, ".csv (", nrow(df), " rows)")

  # Tier profiles: mean of each input and outcome by tier
  if ("efficiency_tier" %in% names(df)) {
    profile_cols <- intersect(c(DEA_INPUTS, "grad_rate_150_4yr", "avg_efficiency_score"), names(df))
    df |>
      group_by(efficiency_tier) |>
      summarise(n = n(),
                across(all_of(profile_cols), ~ round(mean(., na.rm = TRUE), 4)),
                .groups = "drop") |>
      write_csv(file.path(OUTPUT_DIR, paste0("tier_profiles_", suffix, ".csv")))
    message("Saved tier_profiles_", suffix, ".csv")
  }

  invisible(df)
}

# =============================================================================
# 7. Plots
# =============================================================================
make_plots <- function(df, suffix = "full") {
  tryCatch({
    pdf(file.path(OUTPUT_DIR, paste0("efficiency_plots_", suffix, ".pdf")),
        width = 12, height = 5)

    # Score distributions
    score_cols <- intersect(c("dea_ccr_score", "dea_bcc_score", "sfa_cd_score"), names(df))
    score_cols <- score_cols[sapply(score_cols, function(x) sum(!is.na(df[[x]])) > 0)]

    if (length(score_cols) > 0) {
      plots <- lapply(score_cols, function(col) {
        mu <- mean(df[[col]], na.rm = TRUE)
        ggplot(data.frame(x = df[[col]]), aes(x = x)) +
          geom_histogram(bins = 40, fill = "#4C72B0", color = "white", alpha = 0.85) +
          geom_vline(xintercept = mu, color = "red", linetype = "dashed") +
          labs(title = col, x = "Score", y = "Count") +
          theme_minimal(base_size = 9)
      })
      grid.arrange(grobs = plots, ncol = length(plots),
                   top = paste("Efficiency Score Distributions --", suffix))
    }

    # BCC by sector
    if (all(c("dea_bcc_score", "sector") %in% names(df))) {
      p <- df |>
        mutate(sec = recode(as.character(sector), !!!SECTOR_LABELS,
                            .default = as.character(sector))) |>
        filter(!is.na(dea_bcc_score)) |>
        ggplot(aes(x = sec, y = dea_bcc_score, fill = sec)) +
        geom_boxplot(alpha = 0.7, outlier.size = 0.5, show.legend = FALSE) +
        labs(title = paste("DEA-BCC Scores by Sector --", suffix),
             x = NULL, y = "BCC Score") +
        theme_minimal(base_size = 9) +
        theme(axis.text.x = element_text(angle = 25, hjust = 1))
      print(p)
    }

    # Tier bar chart
    if ("efficiency_tier" %in% names(df)) {
      p <- as.data.frame(table(df$efficiency_tier)) |>
        setNames(c("tier", "n")) |>
        ggplot(aes(x = tier, y = n, fill = tier)) +
        geom_col(show.legend = FALSE) +
        geom_text(aes(label = n), vjust = -0.4, size = 3.5) +
        scale_fill_manual(values = c("High-Efficiency" = "#2ca02c",
                                     "Mid-Efficiency"  = "#ff7f0e",
                                     "Low-Efficiency"  = "#d62728")) +
        labs(title = paste("Efficiency Tier Distribution --", suffix),
             x = NULL, y = "Count") +
        theme_minimal(base_size = 9)
      print(p)
    }

    dev.off()
    message("Saved efficiency_plots_", suffix, ".pdf")

  }, error = function(e) {
    message("Plot error: ", e$message)
    tryCatch(dev.off(), error = function(e) invisible(NULL))
  })
}

# =============================================================================
# 8. Run pipeline
# =============================================================================
message(strrep("=", 60))
message("03_efficiency.R")
message(strrep("=", 60))

# Full sample
df_sub <- build_subset(df)
df_sub <- run_dea(df_sub)
df_sub <- run_sfa(df_sub)
df_sub <- classify_tiers(df_sub)
export_results(df_sub, "full")
make_plots(df_sub, "full")

# Sensitivity: exclude for-profits
tryCatch({
  df_nofp <- build_subset(df, exclude_forprofit = TRUE)
  df_nofp <- run_dea(df_nofp)
  df_nofp <- run_sfa(df_nofp)
  df_nofp <- classify_tiers(df_nofp)
  export_results(df_nofp, "no_forprofit")
  make_plots(df_nofp, "no_forprofit")
}, error = function(e) message("Sensitivity run skipped: ", e$message))

message(strrep("=", 60))
message("Efficiency analysis complete. Outputs in: ", OUTPUT_DIR)
message(strrep("=", 60))
