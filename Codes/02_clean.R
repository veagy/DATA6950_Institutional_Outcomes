# 02_clean.R
# Cleans the integrated dataset and engineers composite features for analysis.
#
# Input:  data/processed/master_aligned_raw.csv  (from 01_integrate.R)
# Output: data/processed/analysis_ready.csv
#         docs/data_dictionary.csv
#
# Missing data strategy:
#   - Outcome variables (grad rates, earnings, repayment) are NEVER imputed.
#     They are structurally sparse by design and are prediction targets.
#   - Input/feature variables with < 20% missing: MICE imputation (pmm).
#     Falls back to median imputation if MICE fails to converge.
#   - Input/feature variables with >= 20% missing: median imputation + binary
#     missingness indicator column retained (_missing suffix).
#   - Variables that remain all-NA after imputation are dropped.
#
# Author: Audrey Nichols | DATA 6950 | Spring 2026

library(tidyverse)
library(mice)
library(naniar)
library(FactoMineR)

# -- Paths and settings -------------------------------------------------------
PROC_DIR <- "data/processed"
DOCS_DIR <- "docs"
dir.create(DOCS_DIR, recursive = TRUE, showWarnings = FALSE)

MISS_THRESHOLD <- 0.20   # variables above this get median imputation, not MICE
MICE_ITER      <- 10     # number of MICE iterations (maxit)
SEED           <- 2026

# Variables that should never be imputed or dropped
OUTCOME_VARS <- c(
  "grad_rate_150_4yr", "grad_rate_150_2yr",
  "grad_rate_200_4yr", "grad_rate_200_2yr",
  "median_earnings_6yr", "median_earnings_10yr",
  "loan_repayment_3yr", "completion_rate_sc", "retention_rate_sc"
)

PROTECTED_COLS <- c(
  OUTCOME_VARS,
  "unitid", "input_year", "outcome_year", "outcome_lag",
  "instnm", "sector", "ccbasic", "stabbr", "base_level", "finance_form"
)

# Financial variables to log-transform (right-skewed distributions)
FINANCIAL_VARS <- c(
  "instr_exp_per_fte", "stud_serv_exp_per_fte", "total_exp",
  "sal_instr_avg_9mo", "sal_noninstr_avg", "total_sal_per_fte",
  "median_earnings_6yr", "median_earnings_10yr"
)

# =============================================================================
# 1. Load data
# =============================================================================
message("Loading data...")
df <- read_csv(file.path(PROC_DIR, "master_aligned_raw.csv"), show_col_types = FALSE)
message(nrow(df), " rows, ", ncol(df), " columns")

# =============================================================================
# 2. Type coercion and basic fixes
# =============================================================================

# Rescale rates stored as 0-100 percentages to 0-1 proportions
rate_cols <- intersect(
  c("grad_rate_150_4yr", "grad_rate_150_2yr",
    "grad_rate_200_4yr", "grad_rate_200_2yr",
    "loan_repayment_3yr", "completion_rate_sc", "retention_rate_sc"),
  names(df)
)
for (col in rate_cols) {
  df[[col]] <- suppressWarnings(as.numeric(df[[col]]))
  if (!all(is.na(df[[col]])) && max(df[[col]], na.rm = TRUE) > 1) {
    df[[col]] <- df[[col]] / 100
    message(col, " rescaled from 0-100 to 0-1")
  }
}

# Rescale pell_pct if still on 0-100 scale
if ("pell_pct" %in% names(df)) {
  df$pell_pct <- suppressWarnings(as.numeric(df$pell_pct))
  if (!all(is.na(df$pell_pct)) && max(df$pell_pct, na.rm = TRUE) > 1) {
    df$pell_pct <- df$pell_pct / 100
    message("pell_pct rescaled from 0-100 to 0-1")
  }
}

# Winsorise student-to-faculty ratio at 3x the 99th percentile
if ("stud_fac_ratio" %in% names(df)) {
  cap          <- 3 * quantile(df$stud_fac_ratio, 0.99, na.rm = TRUE)
  n_winsorized <- sum(df$stud_fac_ratio > cap, na.rm = TRUE)
  df$stud_fac_ratio <- pmin(df$stud_fac_ratio, cap, na.rm = TRUE)
  if (n_winsorized > 0) message(n_winsorized, " stud_fac_ratio values winsorized at ", round(cap, 1))
}

# Winsorise stud_serv_exp_per_fte at the 99th percentile
# Strayer University showed values of $100k-$193k/student -- clear reporting anomaly
if ("stud_serv_exp_per_fte" %in% names(df)) {
  cap          <- quantile(df$stud_serv_exp_per_fte, 0.99, na.rm = TRUE)
  n_winsorized <- sum(df$stud_serv_exp_per_fte > cap, na.rm = TRUE)
  df$stud_serv_exp_per_fte <- pmin(df$stud_serv_exp_per_fte, cap, na.rm = TRUE)
  if (n_winsorized > 0) message(n_winsorized, " stud_serv_exp_per_fte values winsorized at $", round(cap))
}

# =============================================================================
# 3. Missing data treatment
# =============================================================================
message("--- Missing data ---")

numeric_cols <- names(df)[sapply(df, is.numeric)]
feature_cols <- setdiff(numeric_cols, PROTECTED_COLS)

miss_pct <- sapply(df[feature_cols], function(x) mean(is.na(x)))

below_thresh  <- names(miss_pct[miss_pct > 0  & miss_pct <  MISS_THRESHOLD])
above_thresh  <- names(miss_pct[miss_pct      >= MISS_THRESHOLD])
complete_cols <- names(miss_pct[miss_pct == 0])

message(length(complete_cols), " feature columns fully observed")
message(length(below_thresh),  " feature columns will be MICE-imputed (< ", MISS_THRESHOLD * 100, "% missing)")
message(length(above_thresh),  " feature columns will get median imputation + missingness indicator (>= ", MISS_THRESHOLD * 100, "% missing)")

if (length(above_thresh) > 0) {
  message("High-missingness features: ", paste(above_thresh, collapse = ", "))
}

# Show outcome missingness for reference -- no action taken on these
outcome_miss <- sapply(df[intersect(OUTCOME_VARS, names(df))], function(x) mean(is.na(x)))
message("\nOutcome variable missingness (informational only -- no imputation):")
print(round(sort(outcome_miss, decreasing = TRUE) * 100, 1))

# -- 3a. Binary missingness indicators ----------------------------------------
cols_with_missing <- names(miss_pct[miss_pct > 0])
for (col in cols_with_missing) {
  df[[paste0(col, "_missing")]] <- as.integer(is.na(df[[col]]))
}
message("\nCreated ", length(cols_with_missing), " missingness indicator columns")

# -- 3b. MICE for features below the threshold --------------------------------
if (length(below_thresh) > 0) {

  # Near-zero-variance columns can cause MICE to fail -- median-fill them instead
  low_var <- sapply(below_thresh, function(col) {
    vals <- df[[col]]
    is.na(sd(vals, na.rm = TRUE)) || sd(vals, na.rm = TRUE) == 0 || length(unique(na.omit(vals))) < 3
  })

  if (any(low_var)) {
    message("Median-filling ", sum(low_var), " near-zero-variance columns before MICE: ",
            paste(below_thresh[low_var], collapse = ", "))
    for (col in below_thresh[low_var]) {
      df[[col]] <- ifelse(is.na(df[[col]]), median(df[[col]], na.rm = TRUE), df[[col]])
    }
    below_thresh <- below_thresh[!low_var]
  }

  if (length(below_thresh) > 0) {
    message("Running MICE on ", length(below_thresh), " columns (maxit = ", MICE_ITER, ")...")

    imp <- tryCatch(
      mice(df[below_thresh], m = 1, maxit = MICE_ITER, method = "pmm",
           seed = SEED, printFlag = FALSE),
      error = function(e) {
        message("MICE failed (", conditionMessage(e), ") -- falling back to median imputation")
        NULL
      }
    )

    if (!is.null(imp)) {
      df[below_thresh] <- complete(imp, 1)
      message("MICE complete")
    } else {
      for (col in below_thresh) {
        df[[col]] <- ifelse(is.na(df[[col]]), median(df[[col]], na.rm = TRUE), df[[col]])
      }
      message("Median fallback applied to ", length(below_thresh), " columns")
    }
  }
}

# -- 3c. Median imputation for features at or above the threshold -------------
if (length(above_thresh) > 0) {
  for (col in above_thresh) {
    df[[col]] <- ifelse(is.na(df[[col]]), median(df[[col]], na.rm = TRUE), df[[col]])
  }
  message("Median imputation applied to ", length(above_thresh), " high-missingness columns")
}

# =============================================================================
# 4. Log transformations and Z-score standardization
# =============================================================================

fin_cols <- intersect(FINANCIAL_VARS, names(df))
for (col in fin_cols) {
  df[[paste0(col, "_log")]] <- log1p(pmax(as.numeric(df[[col]]), 0, na.rm = TRUE))
}
message("log1p applied to ", length(fin_cols), " financial columns")

skip_pattern <- "_missing$|_z$|_log$|^unitid$|^sector$|^ccbasic$|^input_year$|^outcome_year$|^outcome_lag$"
to_scale <- setdiff(
  names(df)[sapply(df, is.numeric)],
  union(PROTECTED_COLS, grep(skip_pattern, names(df), value = TRUE))
)

for (col in to_scale) {
  mu  <- mean(df[[col]], na.rm = TRUE)
  sig <- sd(df[[col]],   na.rm = TRUE)
  if (!is.na(sig) && sig > 0) df[[paste0(col, "_z")]] <- (df[[col]] - mu) / sig
}
message("Z-score standardized ", length(to_scale), " columns (_z suffix added)")

# =============================================================================
# 5. Feature engineering
# =============================================================================
message("--- Feature engineering ---")

safe_feature <- function(name, expr) {
  tryCatch(expr, error = function(e) {
    message("Feature '", name, "' failed (", conditionMessage(e), ") -- skipping")
    df
  })
}

# Resource Concentration Index
# Normalized Shannon entropy of spending shares across instructional and student
# services expenditure. 0 = all spending in one area; 1 = perfectly balanced.
df <- safe_feature("RCI", {
  rci_cols <- intersect(c("instr_exp_per_fte", "stud_serv_exp_per_fte"), names(df))
  if (length(rci_cols) >= 2) {
    mat <- pmax(as.matrix(df[rci_cols]), 0)
    mat[is.na(mat)] <- 0
    totals <- rowSums(mat)
    totals[totals == 0] <- NA
    shares <- mat / totals
    shares[shares == 0] <- NA
    entropy <- -rowSums(shares * log(shares), na.rm = TRUE)
    df$resource_concentration_idx <- entropy / log(length(rci_cols))
    message("RCI computed from: ", paste(rci_cols, collapse = ", "))
  } else {
    message("RCI skipped -- need at least 2 expenditure columns")
  }
  df
})

# Student Support Intensity Score (SSIS)
# Weighted composite of advising, tutoring, and wraparound services per student.
# Falls back to stud_serv_exp_per_fte if sub-components are absent.
df <- safe_feature("SSIS", {
  ssis_weights <- c(advising_exp = 0.40, tutoring_exp = 0.35, wraparound_exp = 0.25)
  available    <- ssis_weights[names(ssis_weights) %in% names(df)]

  if (length(available) > 0 && "total_fte" %in% names(df)) {
    total_w <- sum(available)
    df$student_support_intensity <- Reduce("+", lapply(names(available), function(col) {
      (df[[col]] / if_else(df$total_fte > 0, df$total_fte, NA_real_)) *
        (available[[col]] / total_w)
    }))
    message("SSIS computed from: ", paste(names(available), collapse = ", "))
  } else if ("stud_serv_exp_per_fte" %in% names(df)) {
    df$student_support_intensity <- df$stud_serv_exp_per_fte
    message("SSIS: using stud_serv_exp_per_fte as fallback")
  } else {
    message("SSIS skipped -- no matching expenditure columns found")
  }
  df
})

# Value-Added Proxy
# OLS residual from regressing graduation rate on selectivity and Pell concentration.
# Positive = outperforms selectivity-adjusted expectations; negative = underperforms.
df <- safe_feature("VAP", {
  grad_col <- intersect(c("grad_rate_150_4yr", "grad_rate_150_2yr"), names(df))[1]
  sat_cols <- intersect(c("sat75", "act75"), names(df))

  if (!is.na(grad_col) && "pell_pct" %in% names(df)) {
    if (length(sat_cols) > 0) {
      df$sat_act_composite <- rowMeans(scale(df[sat_cols]), na.rm = TRUE)
      formula <- as.formula(paste(grad_col, "~ sat_act_composite + pell_pct"))
    } else {
      formula <- as.formula(paste(grad_col, "~ pell_pct"))
      message("VAP: no SAT/ACT columns found, using pell-only model")
    }
    mask <- complete.cases(df[all.vars(formula)])
    ols  <- lm(formula, data = df[mask, ])
    df$value_added_proxy <- NA_real_
    df$value_added_proxy[mask] <- residuals(ols)
    message("VAP computed (OLS R2 = ", round(summary(ols)$r.squared, 3), ")")
  } else {
    message("VAP skipped -- graduation rate or pell_pct not found")
  }
  df
})

# Selectivity Composite (PCA)
# First principal component of admission rate (reversed), SAT/ACT 75th percentile,
# and yield rate.
df <- safe_feature("SelectivityComposite", {
  sel_cols <- intersect(c("admit_rate", "sat75", "act75", "yield_rate"), names(df))
  if (length(sel_cols) >= 2) {
    sel_data <- df[sel_cols]
    if ("admit_rate" %in% names(sel_data)) {
      sel_data$admit_rate <- 1 - pmin(pmax(sel_data$admit_rate, 0), 1)
    }
    sel_data <- mutate(sel_data, across(everything(),
                                        ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))
    pca      <- PCA(sel_data, scale.unit = TRUE, ncp = 1, graph = FALSE)
    df$selectivity_composite <- pca$ind$coord[, 1]
    message("Selectivity Composite (PCA PC1): ",
            round(pca$eig[1, "percentage of variance"], 1), "% variance explained")
  } else {
    message("Selectivity Composite skipped -- need at least 2 of: admit_rate, sat75, act75, yield_rate")
  }
  df
})

# Financial Health Index
# Row mean of Z-scored financial sub-components. Higher = financially healthier.
df <- safe_feature("FHI", {
  fhi_parts <- list()
  if (all(c("total_exp", "total_fte") %in% names(df))) {
    exp_per_stu <- df$total_exp / if_else(df$total_fte > 0, df$total_fte, NA_real_)
    fhi_parts$exp_per_stu <- as.numeric(scale(exp_per_stu))
  }
  if ("total_sal_per_fte" %in% names(df)) {
    fhi_parts$sal_per_fte <- as.numeric(scale(df$total_sal_per_fte))
  }
  if (length(fhi_parts) > 0) {
    df$financial_health_idx <- rowMeans(do.call(cbind, fhi_parts), na.rm = TRUE)
    message("Financial Health Index computed from ", length(fhi_parts), " sub-components")
  } else {
    message("FHI skipped -- no sub-components found")
  }
  df
})

# Drop engineered features that are highly redundant with simpler columns
engineered <- intersect(
  c("resource_concentration_idx", "student_support_intensity",
    "value_added_proxy", "selectivity_composite", "financial_health_idx"),
  names(df)
)

if (length(engineered) > 1) {
  cor_mat <- cor(df[engineered], use = "pairwise.complete.obs")
  for (i in seq_len(nrow(cor_mat) - 1)) {
    for (j in seq(i + 1, ncol(cor_mat))) {
      if (!is.na(cor_mat[i, j]) && abs(cor_mat[i, j]) > 0.95) {
        drop_col <- colnames(cor_mat)[j]
        if (drop_col %in% names(df)) {
          df[[drop_col]] <- NULL
          message("Dropped '", drop_col, "' (r = ", round(cor_mat[i, j], 3),
                  " with '", rownames(cor_mat)[i], "') -- redundant")
        }
      }
    }
  }
}

# =============================================================================
# 6. Quality checks
# =============================================================================
message("--- Quality checks ---")

for (col in intersect(rate_cols, names(df))) {
  n_out <- sum(df[[col]] < 0 | df[[col]] > 1, na.rm = TRUE)
  if (n_out > 0) message("WARNING: ", col, " has ", n_out, " values outside [0, 1]")
}

for (col in intersect(c("instr_exp_per_fte", "stud_serv_exp_per_fte", "total_exp"), names(df))) {
  n_neg <- sum(df[[col]] < 0, na.rm = TRUE)
  if (n_neg > 0) message("WARNING: ", col, " has ", n_neg, " negative values")
}

remaining_miss <- sapply(df[feature_cols[feature_cols %in% names(df)]], function(x) sum(is.na(x)))
still_missing  <- remaining_miss[remaining_miss > 0]
if (length(still_missing) > 0) {
  message("Columns still missing after imputation:")
  print(still_missing)
} else {
  message("All feature columns fully observed after imputation")
}

# =============================================================================
# 7. Data dictionary
# =============================================================================
dict <- tibble(
  column      = names(df),
  dtype       = sapply(df, function(x) class(x)[1]),
  pct_missing = round(sapply(df, function(x) mean(is.na(x))) * 100, 1),
  notes       = case_when(
    grepl("_log$",     column) ~ "log1p transform of base column",
    grepl("_z$",       column) ~ "Z-score standardized",
    grepl("_missing$", column) ~ "Binary indicator: 1 = originally missing",
    column == "resource_concentration_idx"  ~ "Normalized Shannon entropy of expenditure shares",
    column == "student_support_intensity"   ~ "Weighted composite of support expenditures per student",
    column == "value_added_proxy"           ~ "OLS residual (grad rate ~ selectivity + pell); positive = outperforms",
    column == "selectivity_composite"       ~ "PCA PC1 of admission rate, SAT/ACT 75th pct, yield rate",
    column == "financial_health_idx"        ~ "Row mean of Z-scored financial health sub-components",
    TRUE ~ ""
  )
)

write_csv(dict, file.path(DOCS_DIR, "data_dictionary.csv"))
message("Data dictionary saved (", nrow(dict), " entries)")

# =============================================================================
# 8. Save output
# =============================================================================
out_path <- file.path(PROC_DIR, "analysis_ready.csv")
write_csv(df, out_path)

message("=================================================================")
message("Cleaning complete.")
message("  ", nrow(df), " rows x ", ncol(df), " columns")
message("  Saved to: ", out_path)
message("=================================================================")
