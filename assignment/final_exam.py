import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import statsmodels.api as sma

#the pca function is written for you, call this from your code to calculate the 1st PC
def pca_function(stdata):
    """Returns the sign identified 1st principal component of a data set.
    input: stdata - a n x t pandas data frame
    output: 1st principal component, standardised to s.d = 1 and
    signed to have the same sign as the cross sectional mean of the variables"""
    factor1_us = sma.PCA(stdata, 1).factors
    factor1 = (factor1_us - factor1_us.mean()) / factor1_us.std()
    sgn = np.sign(pd.concat([stdata.mean(1), factor1], axis=1).corr().iloc[1, 0])
    return factor1 * sgn

#produce your analysis for the following five variables
my_srs = ['sasdate','INDPRO', 'UNRATE', 'PAYEMS', 'CPIAUCSL', 'BUSINVx']

#enter your code below, save this file as ‘final_exam.py’, Zip and upload as instructed

df = pd.read_csv('2021-12.csv')
tmp_df = pd.read_csv("fred_md_desc.csv")
Nber_df = pd.read_csv("NBER_DATES.csv")

def main():
    srs = df[my_srs]
    srs = srs.dropna(how='all').reset_index(drop=True)
    desc_srs = tmp_df[tmp_df['fred'].isin(my_srs)]
    desc_srs = desc_srs.dropna(how='all').reset_index(drop=True)

    transformed_data = pd.DataFrame()

    for feature in srs.columns[1:]:
        tcode = srs.loc[0,feature]
        subset_srs = srs.loc[1:,feature]
        if int(tcode)==5:
            transformed_data[feature]=np.log(subset_srs).diff()
        elif int(tcode)==2:
            transformed_data[feature]=subset_srs.diff()
        elif int(tcode)==6:
            transformed_data[feature]=np.log(subset_srs).diff().diff()
        else:
            transformed_data[feature]=subset_srs

    transformed_data.index=pd.to_datetime(srs.loc[1:,'sasdate'])
    transformed_data = transformed_data[transformed_data.index<='2019-12-31']

    normalized_df=((transformed_data-transformed_data.mean())/transformed_data.std()).fillna(0)

    factor_df = pca_function(normalized_df)

    plt.figure(figsize=(12,8))
    plt.subplot(1,2,1)
    plt.hist(factor_df['comp_0'])
    plt.ylabel("Frequency",fontsize=10)
    plt.title("Histogram Distribution",fontsize=15)
    plt.subplot(1,2,2)
    plt.plot(factor_df['comp_0'])
    plt.title("Time Series of Factor",fontsize=15)

    plt.tight_layout()
    plt.suptitle("Primary Factor (1st Principal Component)",x=0.5,y=1.1,fontsize=20)
    plt.savefig("factor.pdf")
    # plt.show()

    transformed_data_lag_1 = transformed_data.shift(1).fillna(0)
    factor_df_lag1 = factor_df.shift(1)

    fitted_values_df = pd.DataFrame()
    for srs in transformed_data.columns:
        concat_df = pd.concat([transformed_data[srs],transformed_data_lag_1[srs],factor_df_lag1],axis=1).set_axis(['srs','ar1','factor1'],axis=1).dropna()
        model1 = sma.OLS(concat_df['srs'],concat_df).fit()
        predicted = model1.fittedvalues
        fitted_values_df[srs] = predicted

    fitted_values_df.to_csv("fitted_values.csv")
    Nber_df.index=pd.to_datetime(Nber_df.iloc[:,0])
    merged_df = fitted_values_df.merge(Nber_df,left_index=True,right_index=True)
    for srs in transformed_data.columns:
        temp_trans = transformed_data.loc[:,srs]
        temp_pred = merged_df.loc[:,[srs,'0']]
        temp_df = temp_trans.to_frame().merge(temp_pred,left_index=True,right_index=True)
        plt.figure(figsize=(12,8))
        sns.lmplot(x=srs+'_x',y=srs+'_y',data=temp_df,col='0',legend=True)
        plt.suptitle(srs,x=0.5,y=1.1)
        plt.savefig(srs+'.pdf',bbox_inches='tight')

if __name__=="__main__":
    main()