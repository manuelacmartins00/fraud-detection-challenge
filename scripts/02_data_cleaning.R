# 02 — Data cleaning and feature engineering
# Produces data/transactions_clean.csv, the input for the Python models (03–07).
#
# No rows are removed:
# - Identical rows are legitimate repeats. There is no transaction ID or
#   timestamp, only the day, so the same card + IP + amount can appear twice.
# - Zero amounts are all card VERIFICATION transactions. They are flagged.

suppressPackageStartupMessages(library(dplyr))

dir.create("data", showWarnings = FALSE)

raw <- read.csv("fraud_challenge_dataset.csv", stringsAsFactors = FALSE, na.strings = "")
cat("Rows read:", nrow(raw), "\n")

# ---- Names and types ------------------------------------------------------

tx <- raw %>%
  rename(
    merchant_account = Merchant.Account,
    payment_method = Payment.Method,
    currency = Currency,
    amount = Amount,
    shopper_country = Shopper.Country,
    issuer_country = Issuer.Country,
    acquirer_response = Acquirer.Response,
    threeds_directory = X3D.Directory.Response,
    threeds_auth = X3D.Authentication.Response,
    cvc2_response = CVC2.Response,
    raw_acquirer_response = Raw.Acquirer.Response,
    liability_shift = Liability.Shift,
    dispute_reason = Dispute.Reason,
    record_type = Record.Type,
    creation_day = Creation.Day,
    bin_group = BIN_Group,
    ip_token = IP_Token
  ) %>%
  mutate(
    creation_day = as.Date(creation_day),
    liability_shift = liability_shift == "True",
    payment_method = tolower(trimws(payment_method)),
    currency = toupper(trimws(currency))
  )

# ---- Missing values -------------------------------------------------------

cat("\n== Missing values before cleaning\n")
na_counts <- colSums(is.na(tx))
print(na_counts[na_counts > 0])

tx <- tx %>%
  mutate(
    # "-" in the 3DS directory response means no 3DS lookup, same as missing
    threeds_directory = ifelse(threeds_directory %in% "-", NA, threeds_directory),
    shopper_country = coalesce(shopper_country, "UNKNOWN"),
    issuer_country = coalesce(issuer_country, "UNKNOWN"),
    type = coalesce(type, "UNKNOWN"),
    domain = coalesce(domain, "UNKNOWN"),
    liability_shift = coalesce(liability_shift, FALSE)
  )
# 3DS / dispute fields stay NA: missing means "not applicable", not unknown.

# ---- Quality checks -------------------------------------------------------

cat("\n== Checks\n")
cat("Identical rows (kept):", sum(duplicated(raw)), "\n")
cat("Zero amounts:", sum(tx$amount == 0), "| all VERIFICATION:",
    all(tx$type[tx$amount == 0] == "VERIFICATION" | tx$domain[tx$amount == 0] == "VERIFICATION"), "\n")
cat("Negative amounts:", sum(tx$amount < 0), "\n")
cat("Currencies:", n_distinct(tx$currency), "\n")
stopifnot(!anyNA(tx$amount), !anyNA(tx$acquirer_response), !anyNA(tx$bin_group), !anyNA(tx$ip_token))

# ---- Currency conversion --------------------------------------------------
# Amount is in each transaction's own currency. amount_eur uses the rates of
# 2025-11-21 (fx_rates_eur.csv). The models keep using the original amount,
# as in the report.

fx <- read.csv("fx_rates_eur.csv")
stopifnot(all(tx$currency %in% fx$currency))
tx <- tx %>%
  left_join(fx, by = "currency") %>%
  mutate(amount_eur = amount / units_per_eur) %>%
  select(-units_per_eur)

# ---- Labels and features --------------------------------------------------

tx <- tx %>%
  mutate(
    fraud = !is.na(record_type),
    approved = acquirer_response == "APPROVED",
    is_verification = amount == 0
  ) %>%
  add_count(bin_group, name = "bin_freq") %>%
  add_count(ip_token, name = "ip_freq") %>%
  add_count(currency, name = "currency_freq") %>%
  group_by(ip_token) %>% mutate(cards_per_ip = n_distinct(bin_group)) %>% ungroup() %>%
  group_by(bin_group) %>% mutate(ips_per_card = n_distinct(ip_token)) %>% ungroup() %>%
  mutate(
    log_amount = log1p(amount),
    is_usd = as.integer(currency == "USD"),
    is_us = as.integer(shopper_country == "US"),
    is_topup = as.integer(type == "TOPUP"),
    first_time_card = as.integer(bin_freq == 1),
    country_mismatch = as.integer(shopper_country != issuer_country | shopper_country == "UNKNOWN"),
    threeds_y = as.integer(threeds_directory %in% "Y"),
    liability_shift = as.integer(liability_shift)
  ) %>%
  select(-raw_acquirer_response)  # free-text duplicate of acquirer_response

cat("\nRows written:", nrow(tx), "| frauds:", sum(tx$fraud), "\n")
write.csv(tx, "data/transactions_clean.csv", row.names = FALSE)
