
import numpy as np
import pandas as pd

#----------------------------------------------------------------------------
# Pre-process Data 
#----------------------------------------------------------------------------
"""
Preprocessing
1. Remove columns that are not numerical or are missing too much data
2. Remove outliers by adding floors or ceilings to several columns
3. Adjust for skewusing the following methods;
    i. log
    ii. square root
    iii. cube root
    iv. cubed
    v. inverse
4. Normalize with the z-score
Now all data is between -4 and 4 
----------------------------------------------------------------------------
"""
def pre_process(df) -> pd.DataFrame:
    """ 
    This function preprocesses the data before it is used in the model.
    Input:  df - (DataFrame) raw data
    Output: df_norm - (DataFrame) processed data
    """
    features = list(df.columns)

    # Remove non-numerical features or features that are missing data
    for column in ['binnedInc','Geography','PctSomeCol18_24','PctEmployed16_Over', 
                'PctPrivateCoverageAlone', 'studyPerCap']:
        features.remove(column)
        
    # Add ceiling and floor to certain features to eliminate outliers
    df = df[
            (df['MedianAge'] <= 100) & 
            (df['AvgHouseholdSize'] >= 1) & 
            (df['incidenceRate'] < 1000) & 
            (df['popEst2015'] < 10000000)
            ]

    # Lists of features, based on the operation to adjust skew
    
    norm_features = [ # no operation needed
        'TARGET_deathRate', 'MedianAge', 'MedianAgeMale', 
        'MedianAgeFemale', 'PctHS18_24', 'PctHS25_Over', 
        'PctPrivateCoverage', 'PctEmpPrivCoverage', 'PctPublicCoverage', 
        'PctPublicCoverageAlone', 'incidenceRate','PercentMarried',
        'PctMarriedHouseholds']
    fixed_log = [ # log operation used
        'avgAnnCount', 'avgDeathsPerYear', 'medIncome', 
        'popEst2015', 'PctBachDeg25_Over', 'povertyPercent']
    fixed_sqrt = [ # square root operation used
        'PctNoHS18_24', 'PctBachDeg18_24', 
        'PctUnemployed16_Over', 'BirthRate']
    fixed_curt = ['PctBlack', 'PctAsian', 'PctOtherRace'] # cubed root operation used
    fixed_cu = ['PctWhite'] # cubed operation used
    fixed_inv = ['AvgHouseholdSize'] # inversion operation used

    df_sub = pd.DataFrame()
    df_norm = pd.DataFrame()
    
    # 
    skew_adj = {
        "_log": [lambda x : np.log(x), fixed_log],
        "_sqrt": [lambda x : x**(1/2), fixed_sqrt],
        "_curt": [lambda x : x**(1/4), fixed_curt],
        "_cu": [lambda x : x**4, fixed_cu],
        "_inv": [lambda x : x**(-1), fixed_inv],
    }

    for adj,value in skew_adj.items():
        for col in value[1]:
            df.loc[:,col+adj] = df[col].apply(value[0])
            norm_features.append(col+adj)
            
    def z_score(col):
        mean = np.mean(col)
        std = np.std(col)
        z = (col - mean)/std
        return z

    for col in norm_features:
        df_norm[col] = z_score(df[col])
    
    return df_norm