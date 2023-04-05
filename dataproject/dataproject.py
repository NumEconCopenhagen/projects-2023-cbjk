import matplotlib.pyplot as plt
import pandas as pd

def extract_data(barsel_true):
    
    """
    Input:   Dataframe 
    Process: Cut and rename for dataframe 
    Output:  Mother_all and Father_all containing days in parential leave each year 
    """

    # a. Extract varaibles for mother 
    Mother_all = barsel_true[(barsel_true['FARUD'] == 'All fathers, regardless of education') 
                              & (barsel_true['MORUD'] == 'All mothers, regardless of education') 
                              & (barsel_true['TAL'] == 'Mother - days on parental leave (benefits) after birth on average') ]
    # b. Cut data 
    Mother_all = Mother_all[['TID', 'INDHOLD']]

    # c. Extract varaibles for father 
    Father_all = barsel_true[(barsel_true['MORUD'] == 'All mothers, regardless of education') 
                              & (barsel_true['FARUD'] == 'All fathers, regardless of education') 
                              & (barsel_true['TAL'] == 'Father - days on parental leave (benefits) after birth on average') ]
    # d. Cut data 
    Father_all = Father_all[['TID', 'INDHOLD']]
    
    return Mother_all, Father_all


def manipulate_data(df):

    """
    Input:   Dataframe 
    Process: Cut and rename dataframes 
    Output:  Father_educ and Mother_educ to illustrate number of days on parential leave for father and mother 
    """

    # a. Cut the data for the father such that the mother's education is constant
    Father_educ = df[df['MORUD'] == 'All mothers, regardless of education']
    
    # b. Remove variables
    Father_educ = Father_educ[(Father_educ['TAL'] != 'Number of couples') 
                           & (Father_educ['TAL'] != 'Mother - days on parental leave (benefits) after birth on average') 
                           & (Father_educ['TAL'] != 'Mother - days on parental leave (benefits) before birth on average')
                           & (Father_educ['FARUD'] != 'All fathers, regardless of education')]
    
    # c. Cut data further
    Father_educ = Father_educ[['FARUD', 'TID', 'INDHOLD']]
    
    # d. Rename the education variables
    Father_educ['FARUD'] = Father_educ['FARUD'].str.replace('Father ', '').str.title()
    
    # e. Cut the data for the mother such that the father's education is constant
    Mother_educ = df[df['FARUD'] == 'All fathers, regardless of education']
    
    # f. Remove variables
    Mother_educ = Mother_educ[(Mother_educ['TAL'] != 'Number of couples') 
                           & (Mother_educ['TAL'] != 'Father - days on parental leave (benefits) after birth on average') 
                           & (Mother_educ['TAL'] != 'Mother - days on parental leave (benefits) before birth on average') 
                           & (Mother_educ['MORUD'] != 'All mothers, regardless of education')]
    
    # g. Cut data further
    Mother_educ = Mother_educ[['MORUD', 'TID', 'INDHOLD']]
    
    # h. Rename the education variables
    Mother_educ['MORUD'] = Mother_educ['MORUD'].str.replace('Mother ', '').str.title()
    Mother_educ['MORUD'] = Mother_educ['MORUD'].str.replace('Tertirary', 'Tertiary').str.title()
    
    return Father_educ, Mother_educ


def plot_parental_leave(Father_educ, Mother_educ):

    """
    Plot for mother's and father's days on parential leave depending on education 
    """

    # a. Define the figure
    fig = plt.figure()

    # b. Define ax to be the first of the two subplots
    ax = fig.add_subplot(2,1,1)
    # c. Extract the name of each education and plot for each for father's education
    for label, group in Father_educ.groupby(['FARUD']):
        group.plot(x='TID', y='INDHOLD', ax=ax, style='-', label=label)
    # d. Add labels, title and legend
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_ylabel('Days')
    ax.set_title('The number of days of parental leave for father')
    ax.legend().remove()

    # e. Define ax to be the second of the two subplots
    ax1 = fig.add_subplot(2,1,2)
    # f. Extract the name of each education and plot for each for mother's education
    for label, group in Mother_educ.groupby(['MORUD']):
        group.plot(x='TID', y='INDHOLD', ax=ax1, style='-', label=label)
    # g. Add labels, title and legend
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Days')
    ax1.set_title('The number of days of parental leave for mother')
    ax1.legend().remove()

    # h. Create a single legend for both subplots
    handles, labels = ax.get_legend_handles_labels()
    order = ['Lower Secondary', 'Upper Secondary', 'Short Cycle Tertiary', 'Bachelor', 'Master']
    handles = [handles[labels.index(label)] for label in order]
    labels = order
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(1.1,0.5))

    return fig


def calculate_father_share(barsel_true, year):

    """
    Input:   Dataframe and year
    Process: Cut and rename dataframe, and find fathers share of parential leave 
    Output:  Pivot tables for father's conditional share of parential leave 
    """

    # a. Cut data to contain education for mother and father 
    Father_part_educ_1 = barsel_true[(barsel_true['FARUD'] != 'All fathers, regardless of education') 
                               & (barsel_true['MORUD'] != 'All mothers, regardless of education') 
                               & (barsel_true['TAL'] != 'Number of couples')
                               & (barsel_true['TAL'] != 'Mother - days on parental leave (benefits) before birth on average')
                               & (barsel_true['TID'] == year)]
    # b. Cut data 
    Father_part_educ_1 = Father_part_educ_1[['TAL', 'MORUD', 'FARUD', 'INDHOLD']]
    # c. Rename the TAL column
    Father_part_educ_1['TAL'] = Father_part_educ_1['TAL'].str.split(' ').str[0]
    # d. Rename MORUD and FARUD
    Father_part_educ_1['MORUD'] = Father_part_educ_1['MORUD'].str.replace('Mother ', '').str.title()
    Father_part_educ_1['MORUD'] = Father_part_educ_1['MORUD'].str.replace('Tertirary', 'Tertiary').str.title()
    Father_part_educ_1['FARUD'] = Father_part_educ_1['FARUD'].str.replace('Father ', '').str.title()
    # e. Split dataframe into one for mother and father 
    Father_part_educ_1_m = Father_part_educ_1[Father_part_educ_1['TAL'] == 'Mother']
    Father_part_educ_1_f = Father_part_educ_1[Father_part_educ_1['TAL'] == 'Father']
    # f. Merge the two dataframes 
    Father_part_educ_1 = pd.merge(Father_part_educ_1_m, Father_part_educ_1_f, on = ['MORUD', 'FARUD'], how = 'left')
    # g. Cut data 
    Father_part_educ_1 = Father_part_educ_1[['MORUD', 'FARUD', 'INDHOLD_x', 'INDHOLD_y']]
    # h. Rename the columns 
    Father_part_educ_1.columns = ['MORUD', 'FARUD', 'mother', 'father']
    # i. Find fathers share of parential leave 
    Father_part_educ_1['FS'] = Father_part_educ_1['father']/(Father_part_educ_1['father'] + Father_part_educ_1['mother'])*100
    Father_part_educ_1['FS'] = Father_part_educ_1['FS'].round(1)
    # j. Make pivot for fathers share in % 
    Father_part_educ_2020_pivot_procent = Father_part_educ_1.pivot_table(index='MORUD', columns='FARUD', values='FS')
    Father_part_educ_2020_pivot_procent = Father_part_educ_2020_pivot_procent.reindex(['Lower Secondary', 'Upper Secondary', 'Short Cycle Tertiary', 'Bachelor', 'Master'], axis=1)
    Father_part_educ_2020_pivot_procent = Father_part_educ_2020_pivot_procent.reindex(['Lower Secondary', 'Upper Secondary', 'Short Cycle Tertiary', 'Bachelor', 'Master'])
    # k. Make pivot for fathers share in days 
    Father_part_educ_2020_pivot_days = Father_part_educ_1.pivot_table(index='MORUD', columns='FARUD', values='father')
    Father_part_educ_2020_pivot_days = Father_part_educ_2020_pivot_days.reindex(['Lower Secondary', 'Upper Secondary', 'Short Cycle Tertiary', 'Bachelor', 'Master'], axis=1)
    Father_part_educ_2020_pivot_days = Father_part_educ_2020_pivot_days.reindex(['Lower Secondary', 'Upper Secondary', 'Short Cycle Tertiary', 'Bachelor', 'Master'])

    return  Father_part_educ_2020_pivot_procent, Father_part_educ_2020_pivot_days