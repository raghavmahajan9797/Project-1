library(QCA)
sink("qca_results.txt", split = TRUE)

conditions <- c("AICAP", "AIGOV", "SUSSRC", "DIGVIS", "SUPCOLL", "SUPDIV")

raw <- read.csv("raw_dataset.csv", stringsAsFactors = FALSE)
rownames(raw) <- raw$CASEID
excel_fz <- read.csv("fuzzy_dataset_excel.csv", stringsAsFactors = FALSE)

pctl_anchors <- list(
  AICAP   = c(e = 0.500, c = 2.03, i = 4.000),
  AIGOV   = c(e = 0.470, c = 1.14, i = 3.144),
  SUSSRC  = c(e = 0.962, c = 2.85, i = 4.000),
  DIGVIS  = c(e = 0.694, c = 1.90, i = 3.470),
  SUPCOLL = c(e = 0.942, c = 2.36, i = 4.000),
  SUPDIV  = c(e = 0.794, c = 1.93, i = 4.000),
  SCR     = c(e = 0.960, c = 2.43, i = 3.756)
)
rubric_anchors <- setNames(
  lapply(c(conditions, "SCR"), function(v) c(e = 1, c = 2, i = 3)),
  c(conditions, "SCR")
)

calibrate_variant <- function(raw, anchors) {
  fz <- raw[, c("CASEID", "Company", "Year")]
  for (v in names(anchors)) {
    a <- anchors[[v]]
    fz[[v]] <- round(calibrate(
      raw[[v]], type = "fuzzy", method = "direct",
      thresholds = sprintf("e=%f, c=%f, i=%f", a["e"], a["c"], a["i"]),
      logistic = TRUE, idm = 0.95
    ), 6)
  }
  rownames(fz) <- fz$CASEID
  fz
}

to_qca <- function(fz) {
  qca_data <- fz[, c(conditions, "SCR")]
  for (v in c(conditions, "SCR")) {
    half <- which(qca_data[[v]] == 0.5)
    if (length(half) > 0) qca_data[[v]][half] <- 0.501
  }
  qca_data$negSCR <- round(1 - qca_data$SCR, 6)
  qca_data
}

apply_firm_correction <- function(tt, qca_data, raw) {
  crisp <- as.data.frame(lapply(qca_data[conditions], function(x) as.integer(x > 0.5)))
  crisp$CASEID  <- rownames(qca_data)
  crisp$Company <- raw$Company[match(crisp$CASEID, raw$CASEID)]
  crisp$config  <- apply(crisp[conditions], 1, paste, collapse = "")
  firm_counts <- aggregate(Company ~ config, data = crisp, FUN = function(x) length(unique(x)))
  names(firm_counts)[2] <- "n_distinct_firms"
  tt$tt$row_id <- rownames(tt$tt)
  tt_df <- tt$tt
  tt_df$config <- apply(tt_df[conditions], 1, paste, collapse = "")
  tt_df <- merge(tt_df, firm_counts, by = "config", all.x = TRUE)
  tt_df$fails_firm_rule <- ifelse(is.na(tt_df$n_distinct_firms), TRUE, tt_df$n_distinct_firms < 2)
  n_before <- sum(as.character(tt_df$OUT) == "1")
  demote_ids <- tt_df$row_id[as.character(tt_df$OUT) == "1" & tt_df$fails_firm_rule]
  tt$tt$OUT <- as.character(tt$tt$OUT)
  if (length(demote_ids) > 0) tt$tt[demote_ids, "OUT"] <- "?"
  cat(sprintf("Positive rows before/after firm-diversity filter: %d -> %d (demoted: %s)\n",
              n_before, n_before - length(demote_ids),
              ifelse(length(demote_ids) == 0, "none", paste(demote_ids, collapse = ", "))))
  tt
}

run_branch <- function(qca_data, outcome, dir_exp_val, label) {
  cat("\n\n#####################################################################\n")
  cat("### ", label, "\n", sep = "")
  cat("#####################################################################\n")
  tt <- truthTable(qca_data, outcome = outcome, conditions = conditions,
                   incl.cut = 0.80, pri.cut = 0.50, n.cut = 2,
                   show.cases = TRUE, sort.by = "incl")
  print(tt)
  tt <- apply_firm_correction(tt, qca_data, raw)
  cat(sprintf("\n--- Necessity (%s) ---\n", outcome))
  for (v in conditions) {
    cat("\n", v, ":\n", sep = ""); print(pof(v, outcome = outcome, data = qca_data, relation = "necessity"))
    cat("~", v, ":\n", sep = ""); print(pof(paste0("~", v), outcome = outcome, data = qca_data, relation = "necessity"))
  }
  cat(sprintf("\n--- Single-condition sufficiency (%s) ---\n", outcome))
  for (v in conditions) {
    cat("\n", v, ":\n", sep = ""); print(pof(v, outcome = outcome, data = qca_data, relation = "sufficiency"))
    cat("~", v, ":\n", sep = ""); print(pof(paste0("~", v), outcome = outcome, data = qca_data, relation = "sufficiency"))
  }
  dir_exp <- setNames(rep(dir_exp_val, length(conditions)), conditions)
  cat("\n=== COMPLEX solution ===\n")
  print(minimize(tt, include = "", details = TRUE, show.cases = TRUE))
  cat("\n=== PARSIMONIOUS solution ===\n")
  print(minimize(tt, include = "?", details = TRUE, show.cases = TRUE))
  cat("\n=== INTERMEDIATE solution ===\n")
  print(minimize(tt, include = "?", details = TRUE, show.cases = TRUE, dir.exp = dir_exp))
}

fz_pctl   <- calibrate_variant(raw, pctl_anchors)
fz_rubric <- calibrate_variant(raw, rubric_anchors)

cat("\n--- Calibration cross-check (R vs Excel, percentile anchors) ---\n")
chk <- merge(fz_pctl, excel_fz, by = "CASEID", suffixes = c("_R", "_Excel"))
for (v in names(pctl_anchors)) {
  d <- abs(chk[[v]] - chk[[paste0("fz_", v)]])
  cat(sprintf("%-8s max|diff| = %.6f   mean|diff| = %.6f\n", v, max(d), mean(d)))
}

qca_pctl   <- to_qca(fz_pctl)
qca_rubric <- to_qca(fz_rubric)

run_branch(qca_pctl,   "SCR",    1, "Percentile anchors -- outcome SCR")
run_branch(qca_rubric, "SCR",    1, "Rubric anchors -- outcome SCR")
run_branch(qca_pctl,   "negSCR", 0, "Percentile anchors -- outcome ~SCR")
run_branch(qca_rubric, "negSCR", 0, "Rubric anchors -- outcome ~SCR")

sink()
