library(PKNCA)

load("/workspaces/PK-Analysis/data/CP1805/CP1805_conc.rda")
load("/workspaces/PK-Analysis/data/CP1805/CP1805_dose.rda")

conc_df <- CP1805_conc[(CP1805_conc$DAY == 1), ]
dose_df <- CP1805_dose[(CP1805_dose$DAY == 1) & (CP1805_dose$ROUTE == 'Dermal'), ]
conc_df <- conc_df[conc_df$ID %in% dose_df$ID, ]

conc_df$CONC[conc_df$CONC == 500] <- NA
conc_df <- conc_df[!is.na(conc_df$CONC), ] 
sub_cols = c("ID", "DAY")
f = duplicated(dose_df[, sub_cols])
dose_df <- dose_df[!f, ]

conc_df$conc_ngmL <- conc_df$CONC 
conc_df$conc_ngL <- conc_df$conc_ngmL * 1e3
dose_df$dose_ng <- dose_df$DOSE * 1e3

conc_obj <- PKNCAconc(conc_df, conc_ngL~TIME|ID )
dose_obj <- PKNCAdose(dose_df, dose_ng~TIME|ID)

data_obj = PKNCAdata(conc_obj, dose_obj)

results <- pk.nca(data_obj)

res_df <- as.data.frame(results)
res_sum <- as.data.frame(summary(results))
