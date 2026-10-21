# 01 — Exploratory data analysis and statistical testing
# Payment performance (approval rate, decline reasons, daily trend), fraud
# concentration, amounts, and whether the security checks (3D Secure, CVC,
# liability shift) discriminate fraud.
#
# Fraud = transaction with a Record Type (NotificationOfFraud, Chargeback,
# NotificationOfChargeback): 122 cases.
#
# Amount is in the transaction's own currency (43 currencies), so every
# monetary total is converted to EUR with fx_rates_eur.csv (rates of
# 2025-11-21, the middle of the data window).

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(scales)
})

dir.create("outputs/tables", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/figures", recursive = TRUE, showWarnings = FALSE)

tx <- read.csv("fraud_challenge_dataset.csv", stringsAsFactors = FALSE, na.strings = "")
fx <- read.csv("fx_rates_eur.csv")
stopifnot(all(tx$Currency %in% fx$currency))

tx <- tx %>%
  left_join(fx, by = c("Currency" = "currency")) %>%
  mutate(
    amount_eur = Amount / units_per_eur,
    approved = Acquirer.Response == "APPROVED",
    fraud = !is.na(Record.Type),
    Creation.Day = as.Date(Creation.Day)
  ) %>%
  add_count(BIN_Group, name = "bin_freq") %>%
  add_count(IP_Token, name = "ip_freq")

frauds <- filter(tx, fraud)
legit <- filter(tx, !fraud)

# Cramér's V from a contingency table
cramers_v <- function(tab) {
  chi <- suppressWarnings(chisq.test(tab, correct = FALSE))
  v <- sqrt(unname(chi$statistic) / (sum(tab) * (min(dim(tab)) - 1)))
  c(chi2 = unname(chi$statistic), p_value = chi$p.value, cramers_v = v)
}

# =========================================================================
# Payment performance
# =========================================================================

overview <- tx %>%
  summarise(
    transactions = n(),
    first_day = min(Creation.Day),
    last_day = max(Creation.Day),
    approval_rate = mean(approved),
    decline_rate = 1 - mean(approved),
    volume_eur = sum(amount_eur),
    approved_volume_eur = sum(amount_eur[approved]),
    volume_raw_sum = sum(Amount),  # mixed currencies, not a real value
    n_approved = sum(approved),
    confirmed_frauds = sum(fraud)
  )
cat("== Overview\n")
print(as.data.frame(overview))
write.csv(overview, "outputs/tables/01_overview.csv", row.names = FALSE)

declines <- tx %>%
  filter(!approved) %>%
  group_by(Acquirer.Response) %>%
  summarise(transactions = n(), volume_eur = sum(amount_eur), .groups = "drop") %>%
  mutate(share_of_declines = transactions / sum(transactions)) %>%
  arrange(desc(transactions))
cat("\n== Main decline reasons\n")
print(as.data.frame(head(declines, 10)))
write.csv(declines, "outputs/tables/01_decline_reasons.csv", row.names = FALSE)

p_declines <- declines %>%
  slice_head(n = 10) %>%
  ggplot(aes(x = reorder(Acquirer.Response, transactions), y = transactions)) +
  geom_col(fill = "#13386B") +
  coord_flip() +
  labs(title = "Top 10 decline reasons", x = NULL, y = "Transactions") +
  theme_minimal()
ggsave("outputs/figures/01_decline_reasons.png", p_declines, width = 7, height = 4.5, dpi = 150)

daily <- tx %>%
  group_by(Creation.Day) %>%
  summarise(
    transactions = n(),
    approval_rate = mean(approved),
    frauds = sum(fraud),
    fraud_rate = mean(fraud),
    .groups = "drop"
  )
cat("\n== Daily\n")
print(as.data.frame(daily))
write.csv(daily, "outputs/tables/01_daily.csv", row.names = FALSE)

p_daily <- ggplot(daily, aes(Creation.Day, approval_rate)) +
  geom_line(colour = "#13386B") +
  geom_point(colour = "#13386B") +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  labs(title = "Daily approval rate", x = NULL, y = "Approval rate") +
  theme_minimal()
ggsave("outputs/figures/01_daily_approval_rate.png", p_daily, width = 7, height = 4, dpi = 150)

# Where approval could improve: segments with >= 100 transactions
segments <- bind_rows(lapply(
  c("Payment.Method", "Currency", "Shopper.Country", "type", "Merchant.Account"),
  function(col) {
    tx %>%
      group_by(segment = .data[[col]]) %>%
      summarise(transactions = n(), approval_rate = mean(approved), .groups = "drop") %>%
      filter(transactions >= 100) %>%
      mutate(dimension = col, .before = 1)
  }
)) %>% arrange(approval_rate)
cat("\n== Lowest approval rates by segment\n")
print(as.data.frame(head(segments, 10)))
write.csv(segments, "outputs/tables/01_approval_by_segment.csv", row.names = FALSE)

# =========================================================================
# Fraud patterns
# =========================================================================

cat("\n== Confirmed frauds:", nrow(frauds), "\n")
print(table(frauds$Record.Type, frauds$Dispute.Reason))

concentration <- bind_rows(lapply(
  c("Currency", "Shopper.Country", "Issuer.Country", "Payment.Method",
    "type", "domain", "Merchant.Account"),
  function(col) {
    frauds %>%
      count(value = .data[[col]], name = "frauds") %>%
      mutate(share = frauds / sum(frauds), dimension = col, .before = 1) %>%
      arrange(desc(frauds))
  }
))
cat("\n== Fraud concentration\n")
print(as.data.frame(concentration))
write.csv(concentration, "outputs/tables/01_fraud_concentration.csv", row.names = FALSE)

# Amounts: fraud vs legitimate (EUR). Frauds are all in USD.
amounts <- tx %>%
  group_by(group = ifelse(fraud, "fraud", "legitimate")) %>%
  summarise(mean_eur = mean(amount_eur), median_eur = median(amount_eur),
            total_eur = sum(amount_eur), .groups = "drop")
cat("\n== Amounts (EUR)\n")
print(as.data.frame(amounts))
cat("Fraud mean in USD:", round(mean(frauds$Amount), 2), "\n")
cat("Fraud mean as % of legitimate mean:",
    percent(amounts$mean_eur[1] / amounts$mean_eur[2], accuracy = 0.1), "\n")
print(wilcox.test(frauds$amount_eur, legit$amount_eur))
write.csv(amounts, "outputs/tables/01_amounts.csv", row.names = FALSE)

p_amount <- ggplot(tx, aes(amount_eur + 0.01, fill = fraud)) +
  geom_density(alpha = 0.5) +
  scale_x_log10(labels = label_number()) +
  scale_fill_manual(values = c(`FALSE` = "#9AA5B1", `TRUE` = "#00C2C7"),
                    labels = c("Legitimate", "Fraud"), name = NULL) +
  labs(title = "Amount distribution (EUR, log scale)", x = "Amount (EUR)", y = "Density") +
  theme_minimal()
ggsave("outputs/figures/01_amount_distribution.png", p_amount, width = 7, height = 4, dpi = 150)

# BIN / IP reuse among frauds
velocity <- frauds %>%
  count(BIN_Group, bin_freq, IP_Token, ip_freq, name = "frauds") %>%
  arrange(desc(frauds))
cat("\n== Fraud BIN / IP groups\n")
print(as.data.frame(velocity))
write.csv(velocity, "outputs/tables/01_fraud_velocity.csv", row.names = FALSE)

# =========================================================================
# Security checks: statistical testing
# =========================================================================
# 3DS directory: Y = authenticated, A = attempted. CVC2 code 2 = no match.

cat("\n== Security checks among frauds\n")
cat("Passed 3D Secure (Y/A):", percent(mean(frauds$X3D.Directory.Response %in% c("Y", "A")), 0.1), "\n")
cat("CVC not failed:        ", percent(mean(frauds$CVC2.Response != 2), 0.1), "\n")
cat("Liability shift:       ", percent(mean(frauds$Liability.Shift %in% "True"), 0.1), "\n")

as_cat <- function(x) ifelse(is.na(x), "missing", as.character(x))
security <- list(
  threeds_directory = as_cat(tx$X3D.Directory.Response),
  threeds_auth = as_cat(tx$X3D.Authentication.Response),
  cvc2_response = as_cat(tx$CVC2.Response),
  liability_shift = as_cat(tx$Liability.Shift)
)
security$threeds_x_cvc <- paste(security$threeds_directory, security$cvc2_response)
security$threeds_x_liability <- paste(security$threeds_directory, security$liability_shift)
security$all_security <- do.call(paste, security[1:4])

tests <- as.data.frame(t(sapply(security, function(x) cramers_v(table(x, tx$fraud)))))
tests$feature <- rownames(tests)
tests <- tests[order(-tests$cramers_v), c("feature", "chi2", "p_value", "cramers_v")]
cat("\n== Chi-square / Cramér's V (security features vs fraud)\n")
print(tests, row.names = FALSE)
write.csv(tests, "outputs/tables/01_security_tests.csv", row.names = FALSE)
