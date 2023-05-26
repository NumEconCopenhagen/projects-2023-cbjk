import matplotlib.pyplot as plt
import pandas as pd
from  matplotlib.colors import LinearSegmentedColormap

def numb_parent(barsel_true):
    """ 
    Cut dataframe for number of parents
    """

    # a. cut data 
    number = barsel_true[(barsel_true['Numb'] == 'Number of couples')
                        & (barsel_true['Mom_educ'] == 'All mothers, regardless of education')
                        & (barsel_true['Dad_educ'] == 'All fathers, regardless of education')]
    
    # b. return the dataframe 
    return number



def extract_data(barsel_true):
    
    """
    Input:   Dataframe 
    Process: Cut and rename for dataframe 
    Output:  Mother_all and Father_all containing days in parential leave each year 
    """

    # a. Extract varaibles for mother 
    Mother_all = barsel_true[(barsel_true['Dad_educ'] == 'All fathers, regardless of education') 
                              & (barsel_true['Mom_educ'] == 'All mothers, regardless of education') 
                              & (barsel_true['Numb'] == 'Mother - days on parental leave (benefits) after birth on average') ]
    # b. Cut data 
    Mother_all = Mother_all[['Year', 'Count']]

    # c. Extract varaibles for father 
    Father_all = barsel_true[(barsel_true['Mom_educ'] == 'All mothers, regardless of education') 
                              & (barsel_true['Dad_educ'] == 'All fathers, regardless of education') 
                              & (barsel_true['Numb'] == 'Father - days on parental leave (benefits) after birth on average') ]
    # d. Cut data 
    Father_all = Father_all[['Year', 'Count']]
    
    return Mother_all, Father_all


def cut_data(df):

    """
    Input:   Dataframe 
    Process: Cut and rename dataframes 
    Output:  Father_educ and Mother_educ to illustrate number of days on parential leave for father and mother 
    """

    # a. Cut the data for the father such that the mother's education is constant
    Father_educ = df[df['Mom_educ'] == 'All mothers, regardless of education']
    
    # b. Remove variables
    Father_educ = Father_educ[(Father_educ['Numb'] != 'Number of couples') 
                           & (Father_educ['Numb'] != 'Mother - days on parental leave (benefits) after birth on average') 
                           & (Father_educ['Numb'] != 'Mother - days on parental leave (benefits) before birth on average')
                           & (Father_educ['Dad_educ'] != 'All fathers, regardless of education')]
    
    # c. Cut data further
    Father_educ = Father_educ[['Dad_educ', 'Year', 'Count']]
    
    # d. Rename the education variables
    Father_educ['Dad_educ'] = Father_educ['Dad_educ'].str.replace('Father ', '').str.title()
    
    # e. Cut the data for the mother such that the father's education is constant
    Mother_educ = df[df['Dad_educ'] == 'All fathers, regardless of education']
    
    # f. Remove variables
    Mother_educ = Mother_educ[(Mother_educ['Numb'] != 'Number of couples') 
                           & (Mother_educ['Numb'] != 'Father - days on parental leave (benefits) after birth on average') 
                           & (Mother_educ['Numb'] != 'Mother - days on parental leave (benefits) before birth on average') 
                           & (Mother_educ['Mom_educ'] != 'All mothers, regardless of education')]
    
    # g. Cut data further
    Mother_educ = Mother_educ[['Mom_educ', 'Year', 'Count']]
    
    # h. Rename the education variables
    Mother_educ['Mom_educ'] = Mother_educ['Mom_educ'].str.replace('Mother ', '').str.title()
    Mother_educ['Mom_educ'] = Mother_educ['Mom_educ'].str.replace('Tertirary', 'Tertiary').str.title()
    
    return Father_educ, Mother_educ


def plot_parental_leave(Father_educ, Mother_educ):

    """
    Plot for mother's and father's days on parential leave depending on education 
    """

    # a. Define the figure
    fig = plt.figure()

    # b. Define ax to be the first of the two subplots
    ax = fig.add_subplot(2,1,2)
    # c. Extract the name of each education and plot for each for father's education
    for label, group in Father_educ.groupby(['Dad_educ']):
        group.plot(x='Year', y='Count', ax=ax, style='-', label=label)
    # d. Add labels, title and legend
    ax.set_xlabel('Year')
    ax.set_ylabel('Days')
    ax.set_title('The number of days of parental leave for mother')
    ax.legend().remove()

    # e. Define ax to be the second of the two subplots
    ax1 = fig.add_subplot(2,1,1)
    # f. Extract the name of each education and plot for each for mother's education
    for label, group in Mother_educ.groupby(['Mom_educ']):
        group.plot(x='Year', y='Count', ax=ax1, style='-', label=label)
    # g. Add labels, title and legend
    ax1.set_xlabel('')
    ax1.set_xticklabels([])
    ax1.set_ylabel('Days')
    ax1.set_title('The number of days of parental leave for father')
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
    Father_part_educ_1 = barsel_true[(barsel_true['Dad_educ'] != 'All fathers, regardless of education') 
                               & (barsel_true['Mom_educ'] != 'All mothers, regardless of education') 
                               & (barsel_true['Numb'] != 'Number of couples')
                               & (barsel_true['Numb'] != 'Mother - days on parental leave (benefits) before birth on average')
                               & (barsel_true['Year'] == year)]
    
    # b. Cut data 
    Father_part_educ_1 = Father_part_educ_1[['Numb', 'Mom_educ', 'Dad_educ', 'Count']]
    
    # c. Rename the TAL column
    Father_part_educ_1['Numb'] = Father_part_educ_1['Numb'].str.split(' ').str[0]
    
    # d. Rename MORUD and FARUD
    Father_part_educ_1['Mom_educ'] = Father_part_educ_1['Mom_educ'].str.replace('Mother ', '').str.title()
    Father_part_educ_1['Mom_educ'] = Father_part_educ_1['Mom_educ'].str.replace('Tertirary', 'Tertiary').str.title()
    Father_part_educ_1['Dad_educ'] = Father_part_educ_1['Dad_educ'].str.replace('Father ', '').str.title()
    
    # e. Split dataframe into one for mother and father 
    Father_part_educ_1_m = Father_part_educ_1[Father_part_educ_1['Numb'] == 'Mother']
    Father_part_educ_1_f = Father_part_educ_1[Father_part_educ_1['Numb'] == 'Father']
    
    # f. Merge the two dataframes 
    Father_part_educ_1 = pd.merge(Father_part_educ_1_m, Father_part_educ_1_f, on = ['Mom_educ', 'Dad_educ'], how = 'left')
    
    # g. Cut data 
    Father_part_educ_1 = Father_part_educ_1[['Mom_educ', 'Dad_educ', 'Count_x', 'Count_y']]
    
    # h. Rename the columns 
    Father_part_educ_1.columns = ['Mom_educ', 'Dad_educ', 'mother', 'father']
    
    # i. Find fathers share of parential leave 
    Father_part_educ_1['FS'] = Father_part_educ_1['father']/(Father_part_educ_1['father'] + Father_part_educ_1['mother'])*100
    Father_part_educ_1['FS'] = Father_part_educ_1['FS'].round(1)
    
    # j. Make pivot for fathers share in % 
    Father_part_educ_2020_pivot_procent = Father_part_educ_1.pivot_table(index='Mom_educ', columns='Dad_educ', values='FS').round(1)
    Father_part_educ_2020_pivot_procent = Father_part_educ_2020_pivot_procent.reindex(['Lower Secondary', 'Upper Secondary', 'Short Cycle Tertiary', 'Bachelor', 'Master'], axis=1)
    Father_part_educ_2020_pivot_procent = Father_part_educ_2020_pivot_procent.reindex(['Lower Secondary', 'Upper Secondary', 'Short Cycle Tertiary', 'Bachelor', 'Master'])
    
    # k. Make pivot for fathers share in days 
    Father_part_educ_2020_pivot_days = Father_part_educ_1.pivot_table(index='Mom_educ', columns='Dad_educ', values='father').round(1)
    Father_part_educ_2020_pivot_days = Father_part_educ_2020_pivot_days.reindex(['Lower Secondary', 'Upper Secondary', 'Short Cycle Tertiary', 'Bachelor', 'Master'], axis=1)
    Father_part_educ_2020_pivot_days = Father_part_educ_2020_pivot_days.reindex(['Lower Secondary', 'Upper Secondary', 'Short Cycle Tertiary', 'Bachelor', 'Master'])
    
    # l. color the pivots 
    # i. set colors
    cmap = LinearSegmentedColormap.from_list('rg',["r", "yellow", "g"], N=256) 
    # ii. color the procent pivot 
    Father_part_educ_2020_pivot_procent_color = Father_part_educ_2020_pivot_procent.style.background_gradient(
        cmap = cmap, axis = None)
    # ii. color the days pivot 
    Father_part_educ_2020_pivot_days_color = Father_part_educ_2020_pivot_days.style.background_gradient(
        cmap = cmap, axis = None)
    
    # m. set precision for the pivot tables
    Father_part_educ_2020_pivot_procent_color = Father_part_educ_2020_pivot_procent_color.format(precision=1)
    Father_part_educ_2020_pivot_days_color = Father_part_educ_2020_pivot_days_color.format(precision=1)

    return Father_part_educ_2020_pivot_procent_color, Father_part_educ_2020_pivot_days_color