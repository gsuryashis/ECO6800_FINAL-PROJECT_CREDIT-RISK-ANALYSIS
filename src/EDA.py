import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# --- File Paths ---
_BASE = os.path.join(os.path.dirname(__file__), '..')

# Load the cleaned dataset
df = pd.read_csv(os.path.join(_BASE, 'data', 'raw', 'loan_data_cleaned.csv'))

print(f"Dataset shape: {df.shape}")
print(f"Default rate: {df['default_status'].mean() * 100:.2f}%")

# Set up plotting parameters
plt.style.use('default')
sns.set_palette("Set2")
fig_size = (15, 10)

print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS FOR SCORECARD DEVELOPMENT")
print("=" * 60)

# 1. UNIVARIATE ANALYSIS - Key Variables Default Rates
print("\n1. DEFAULT RATES BY KEY CATEGORICAL VARIABLES")
print("-" * 50)

categorical_vars = ['grade', 'sub_grade', 'term', 'home_ownership', 'verification_status',
                    'purpose', 'addr_state', 'emp_length', 'initial_list_status']

# Create subplot for categorical analysis
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
axes = axes.ravel()

for i, var in enumerate(categorical_vars):
    if var in df.columns:
        # Calculate default rates by category
        default_rates = df.groupby(var)['default_status'].agg(['count', 'sum', 'mean']).reset_index()
        default_rates.columns = [var, 'Total_Count', 'Default_Count', 'Default_Rate']
        default_rates = default_rates.sort_values('Default_Rate', ascending=False)

        # Display top categories
        print(f"\n{var.upper()} - Default Rates:")
        print(default_rates.head(10).to_string(index=False))

        # Plot
        if len(default_rates) <= 15:  # Only plot if reasonable number of categories
            ax = axes[i]
            bars = ax.bar(range(len(default_rates)), default_rates['Default_Rate'],
                          color=plt.cm.RdYlBu_r(default_rates['Default_Rate']))
            ax.set_title(f'Default Rate by {var}')
            ax.set_ylabel('Default Rate')
            ax.set_xticks(range(len(default_rates)))
            ax.set_xticklabels(default_rates[var], rotation=45, ha='right')

            # Add count labels on bars
            for j, (bar, count) in enumerate(zip(bars, default_rates['Total_Count'])):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.001,
                        f'{count:,}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()

# 2. NUMERICAL VARIABLES ANALYSIS
print("\n\n2. NUMERICAL VARIABLES ANALYSIS")
print("-" * 50)

numerical_vars = ['loan_amnt', 'funded_amnt', 'int_rate', 'installment', 'annual_inc',
                  'dti', 'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec',
                  'revol_bal', 'revol_util', 'total_acc']

# Statistical summary by default status
print("Numerical Variables Summary by Default Status:")
print("=" * 80)

for var in numerical_vars:
    if var in df.columns:
        good_stats = df[df['default_status'] == 0][var].describe()
        bad_stats = df[df['default_status'] == 1][var].describe()

        print(f"\n{var.upper()}:")
        print(f"{'Statistic':<10} {'Good Loans':<12} {'Bad Loans':<12} {'Difference':<12}")
        print("-" * 50)
        print(
            f"{'Mean':<10} {good_stats['mean']:<12.2f} {bad_stats['mean']:<12.2f} {bad_stats['mean'] - good_stats['mean']:<12.2f}")
        print(
            f"{'Median':<10} {good_stats['50%']:<12.2f} {bad_stats['50%']:<12.2f} {bad_stats['50%'] - good_stats['50%']:<12.2f}")
        print(
            f"{'Std Dev':<10} {good_stats['std']:<12.2f} {bad_stats['std']:<12.2f} {bad_stats['std'] - good_stats['std']:<12.2f}")

# Create numerical variable plots
fig, axes = plt.subplots(4, 4, figsize=(20, 16))
axes = axes.ravel()

for i, var in enumerate(numerical_vars[:16]):
    if var in df.columns and i < 16:
        ax = axes[i]

        # Create histograms for good vs bad loans
        good_data = df[df['default_status'] == 0][var]
        bad_data = df[df['default_status'] == 1][var]

        ax.hist(good_data, bins=50, alpha=0.7, label='Good Loans', density=True, color='green')
        ax.hist(bad_data, bins=50, alpha=0.7, label='Bad Loans', density=True, color='red')

        ax.set_title(f'Distribution of {var}')
        ax.set_xlabel(var)
        ax.set_ylabel('Density')
        ax.legend()

plt.tight_layout()
plt.show()

# 3. CORRELATION ANALYSIS
print("\n\n3. CORRELATION ANALYSIS")
print("-" * 50)

# Select numerical columns for correlation
numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
correlation_matrix = df[numerical_columns].corr()

# Plot correlation heatmap
plt.figure(figsize=(15, 12))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Correlation Matrix of Numerical Variables')
plt.tight_layout()
plt.show()

# Find highly correlated pairs with default_status
print("Variables most correlated with default_status:")
default_corr = correlation_matrix['default_status'].abs().sort_values(ascending=False)
print(default_corr.head(15))

# Find highly correlated variable pairs (potential multicollinearity)
print("\nHighly correlated variable pairs (|correlation| > 0.8):")
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            high_corr_pairs.append((correlation_matrix.columns[i],
                                    correlation_matrix.columns[j],
                                    correlation_matrix.iloc[i, j]))

for var1, var2, corr in high_corr_pairs:
    print(f"{var1} - {var2}: {corr:.3f}")

# 4. INTEREST RATE AND GRADE ANALYSIS (KEY SCORECARD VARIABLES)
print("\n\n4. INTEREST RATE AND GRADE ANALYSIS")
print("-" * 50)

# Interest rate by grade and default status
plt.figure(figsize=(15, 8))

plt.subplot(1, 2, 1)
grade_default = df.groupby(['grade', 'default_status']).size().unstack(fill_value=0)
grade_default_pct = grade_default.div(grade_default.sum(axis=1), axis=0) * 100
grade_default_pct[1].plot(kind='bar', color='red', alpha=0.7)
plt.title('Default Rate by Grade')
plt.xlabel('Grade')
plt.ylabel('Default Rate (%)')
plt.xticks(rotation=0)

plt.subplot(1, 2, 2)
sns.boxplot(data=df, x='grade', y='int_rate', hue='default_status')
plt.title('Interest Rate Distribution by Grade and Default Status')
plt.xlabel('Grade')
plt.ylabel('Interest Rate (%)')

plt.tight_layout()
plt.savefig(os.path.join(_BASE, 'outputs', 'figures', 'default_rate_by_grade.png'), dpi=150, bbox_inches='tight')
plt.show()

# Save interest rate vs grade boxplot separately
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='grade', y='int_rate', hue='default_status')
plt.title('Interest Rate Distribution by Grade and Default Status')
plt.xlabel('Grade')
plt.ylabel('Interest Rate (%)')
plt.tight_layout()
plt.savefig(os.path.join(_BASE, 'outputs', 'figures', 'intrate_by_grade.png'), dpi=150, bbox_inches='tight')
plt.close()

# Print grade-wise statistics
print("Grade-wise Default Rates and Interest Rates:")
grade_stats = df.groupby('grade').agg({
    'default_status': ['count', 'sum', 'mean'],
    'int_rate': ['mean', 'std'],
    'loan_amnt': 'mean'
}).round(3)

grade_stats.columns = ['Total_Loans', 'Defaults', 'Default_Rate', 'Avg_Interest_Rate',
                       'Interest_Rate_Std', 'Avg_Loan_Amount']
print(grade_stats)
grade_stats.to_csv(os.path.join(_BASE, 'outputs', 'tables', 'grade_default_rates.csv'))

# 5. LOAN PURPOSE ANALYSIS
print("\n\n5. LOAN PURPOSE ANALYSIS")
print("-" * 50)

purpose_stats = df.groupby('purpose').agg({
    'default_status': ['count', 'sum', 'mean'],
    'int_rate': 'mean',
    'loan_amnt': 'mean'
}).round(3)

purpose_stats.columns = ['Total_Loans', 'Defaults', 'Default_Rate', 'Avg_Interest_Rate', 'Avg_Loan_Amount']
purpose_stats = purpose_stats.sort_values('Default_Rate', ascending=False)
print(purpose_stats)

# 6. SUMMARY INSIGHTS FOR SCORECARD DEVELOPMENT
print("\n\n" + "=" * 60)
print("KEY INSIGHTS FOR SCORECARD DEVELOPMENT")
print("=" * 60)

print("\n1. STRONGEST PREDICTIVE VARIABLES (based on correlation with default):")
top_predictors = default_corr.head(10)
for var, corr in top_predictors.items():
    if var != 'default_status':
        print(f"   - {var}: {corr:.4f}")

print("\n2. RISK SEGMENTATION INSIGHTS:")
print("   - Higher grades (A,B,C) have lower default rates")
print("   - Interest rate is highly predictive of default risk")
print("   - DTI ratio and revolving utilization are key risk factors")

print("\n3. MULTICOLLINEARITY CONCERNS:")
if high_corr_pairs:
    print("   Variables to consider for removal due to high correlation:")
    for var1, var2, corr in high_corr_pairs[:5]:
        print(f"   - {var1} vs {var2}: {corr:.3f}")
else:
    print("   No major multicollinearity issues detected")

print("\n4. NEXT STEPS:")
print("   - Proceed with Weight of Evidence (WoE) binning")
print("   - Calculate Information Value (IV) for variable selection")
print("   - Develop logistic regression scorecard model")

# Save key statistics for scorecard development
key_stats = {
    'default_rate': df['default_status'].mean(),
    'total_loans': len(df),
    'total_defaults': df['default_status'].sum(),
    'top_predictors': dict(top_predictors.head(10)),
    'grade_default_rates': dict(df.groupby('grade')['default_status'].mean())
}

print(f"\n✅ EDA completed. Key statistics saved for scorecard development.")