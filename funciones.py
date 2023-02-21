import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests, json
import re
from collections import Counter
from IPython.core.display import display, HTML
from tqdm import tqdm

from scipy.stats import median_abs_deviation
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

######## FUNCIONES #########

def calidad_datos(data):
    '''
    Función que entrega información del dataset (compartida por José Terrazas)
    
    Parámetros:
        No tiene
        
    Retorno:
        Por cada variable (columna) del dataset, informa:
        Tipo de datos, cantidad de valores nulos, porcentaje de nulos, cantidad de ceros, 
        porcentaje de ceros, total de datos, total de valores únicos, valor más frecuente, 
        cantidad de valores más frecuentes, media, desviación estándar
    '''
    tipos = pd.DataFrame({'tipo': data.dtypes},index=data.columns)
    na = pd.DataFrame({'nulos': data.isna().sum()}, index=data.columns)
    na_prop = pd.DataFrame({'porc_nulos':data.isna().sum()/data.shape[0]},index=data.columns)
    ceros = pd.DataFrame({'ceros':[data.loc[data[col]==0,col].shape[0] for col in data.columns]},index= data.columns)
    ceros_prop = pd.DataFrame({'porc_ceros':[data.loc[data[col]==0,col].shape[0]/data.shape[0] for col in data.columns]},index= data.columns)

    summary = data.describe(include='all').T
    summary['dist_IQR'] = summary['75%'] - summary['25%']
    summary['limit_inf'] = summary['25%'] - summary['dist_IQR']*1.5
    summary['limit_sup'] = summary['75%'] + summary['dist_IQR']*1.5
    summary['outliers'] = data.apply(lambda x: sum(np.where((x<summary['limit_inf'][x.name]) | (x>summary['limit_sup'][x.name]),1 ,0)) if x.name in summary['limit_inf'].dropna().index else 0)

    return pd.concat([tipos, na, na_prop, ceros, ceros_prop, summary], axis=1).sort_values('tipo')

def limpieza(data, df_init):
    '''
    Función que limpia y ordena el dataset. 
    También deja nombres de columnas y datos en minúsculas, y desglosa datos relevantes del atributo Vin
    
    Parámetros:
        data: dataset que se requiere limpiar
        df_init: dataset original
        
    Retorno:
        Dataset "limpio"
    '''
    # Nombres de columnas en minuscula
    data.columns=data.columns.str.lower()
    df_init.columns=df_init.columns.str.lower()

    #Estandarizar datos de variables categóricas con minúsculas
    # Pasar a minusculas y limpiar leading spaces
    vars_to_clean = ['state','make','model','city']
    data[vars_to_clean] = df_init[vars_to_clean].apply(lambda x: x.str.lower()).apply(lambda x: x.str.strip())

    # Reemplazando , por / en 2 registros
    data['model']=data['model'].replace('RAV4V6,Sport,JBL', 'RAV4V6/Sport/JBL')
    data['model']=data['model'].replace('SiennaXLE,CERTIFIED,', 'SiennaXLE/CERTIFIED,')

    # Reemplazando Model '3-Sep' por '9-3' en 2 registros
    data['model']=data['model'].replace('3-Sep', '9-3')

    #Eliminando 12.682 registros que tienen ',' al final en Model 
    data['model']=data['model'].replace(',','')
    
    # Solo quedarse con datos que tienen Vin de largo 17 
    data = data[data.vin.str.len() == 17]

    #Expansión de Vin
    data['pais_origen']       = data['vin'].str[0:1]
    data['categoria']         = data['vin'].str[2:3]
    # data['digito_conf']     = data['vin'].str[8:9]
    data['planta']            = data['vin'].str[10:11]
    data=data.drop(columns='vin')

    # Renombrar columnas para mejor lectura
    data.rename(columns={'year':'anio_fabricacion','price':'precio','city':'ciudad_transaccion','state':'estado','mileage':'kilometraje','make':'fabricante','model':'modelo'}, inplace=True)

    return data

def replacements_batch(x, list_replacements):
    '''
    Función que reemplaza x por valor que corresponde de un diccionario.
    Se busca marca en diccionario y se reemplaza valor de x por valor que corresponde.

    Parámetros:
        x: valor que se requiere reemplazar
        list_replacements: diccionario con valores posibles por marca de vehículo
        
    Retorno:
        x
    '''    
    for reps in list_replacements:
        x_new = re.sub(reps[0],reps[1],x)
        if x_new != x:
            return x_new
    return x

def binarizarAtributo(df, atributo, new_atributo, bins, estrategia):
    '''
    Función que agrupa datos continuos en intervalos, para un atributo (columna) determinado.
    A esto le llamamos binarizar.
    Parámetros:
        df: dataset donde se encuentra el atributo
        atributo: atributo con datos a binarizar
        new_atributo: nuevo atributo en el que se registrarán los datos binarizados asociados al atributo original 
        bins: número de intervalos a generar
        estrategia: estrategia utilizada para definir la anchura de los intervalos
    Retorno:
        Ajuste y escalamiento de los datos de columna atributo.
    '''
    ct = ColumnTransformer([(new_atributo, KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy=estrategia), [0])])
    ct.fit(df[[atributo]])
    return ct


def binarizaAtributo(df, atributo, new_atributo, bins, estrategia):
    '''
    Función que agrupa datos continuos en intervalos, para un atributo (columna) determinado.
    A esto le llamamos binarizar.
    Parámetros:
        df: dataset donde se encuentra el atributo
        atributo: atributo con datos a binarizar
        new_atributo: nuevo atributo en el que se registrarán los datos binarizados asociados al atributo original 
        bins: número de intervalos a generar
        estrategia: estrategia utilizada para definir la anchura de los intervalos
    Retorno:
        Ajuste y escalamiento de los datos de columna atributo.
    '''    
    ct = ColumnTransformer([(new_atributo, KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy=estrategia), [0])])
    return ct.fit_transform(df[[atributo]]).T[0]

# def binarizaAtributo(df, atributo, new_atributo, bins, estrategia, entrena_fit=False):
#     '''
#     Función que agrupa datos continuos en intervalos, para un atributo (columna) determinado.
#     A esto le llamamos binarizar.

#     Parámetros:
#         df: dataset donde se encuentra el atributo
#         atributo: atributo con datos a binarizar
#         new_atributo: nuevo atributo en el que se registrarán los datos binarizados asociados al atributo original 
#         bins: número de intervalos a generar
#         estrategia: estrategia utilizada para definir la anchura de los intervalos

#     Retorno:
#         Ajuste y escalamiento de los datos de columna atributo.
#     '''    
#     ct = ColumnTransformer([(new_atributo, KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy=estrategia), [0])])

#     if entrena_fit:
#         print("por aqui")
#         ct.fit(df[[atributo]])
#     else:
#         ct.fit_transform(df[[atributo]]).T[0]   
    # return ct

def recodifica_modelos(df, marcas, col_new, col_marca, col_modelo):
    for marca in marcas:
        if marca in lista_cambios.keys():
            df.loc[col_marca == marca,col_new] = df.loc[col_marca == marca,col_modelo].apply(lambda x: replacements_batch(x, lista_cambios[marca]))
    else:
        df.loc[col_marca == marca,col_new] = df.loc[col_marca == marca, col_modelo]

    return df    

def cuentas_por_grupo(df, grupos, variables_a_agrupar, n_bins, n_threshold):
    '''
    Función que determina la cantidad de observaciones de la muestra que se asocian a un grupo.

    Parámetros:
        df: 
        grupos: 
        variables_a_agrupar: 
        n_bins: 
        n_threshold: 

    Retorno:
        groups_final_ordenado: número de observaciones (n) asociadas a cada grupo
        grupos_reducidos: grupos
    '''
    groups_final = []
    for tupla in grupos.indices.keys():
        groups_final.append([*tupla, len(grupos.get_group(tupla))])
    groups_final = pd.DataFrame(groups_final, columns=[*variables_a_agrupar, 'n'])
    display(HTML(
        f'<p style="font-size:20px;font-weight:bold">En la data entregada, se obtuvo un total de <span style="color:tomato">{len(groups_final)}</span> grupos, agrupando según las variables <span style="color:tomato">{", ".join(variables_a_agrupar)}</span>. De ese total de grupos, sólo <span style="color:tomato">{len(groups_final[groups_final.n > n_threshold])}</span> tienen más de <span style="color:tomato">{n_threshold}</span> datos.</p>'))
    groups_final_ordenado = groups_final.sort_values(by=["n"], ascending=False).reset_index(drop=True)
    display(HTML('<p style="font-size:20px;font-weight:bold">Lista de grupos ordenados por cantidad de datos</p>'))
    display(groups_final_ordenado.loc[groups_final_ordenado.n > n_threshold])
    groups_final[groups_final.n > n_threshold].hist(column='n', bins=n_bins)
    plt.xlabel('Número de datos presente en el grupo')
    plt.ylabel('Cantidad de grupos')
    plt.title('Cantidad de grupos por número de datos.')

    groups_to_remove = groups_final_ordenado.loc[~(groups_final_ordenado.n > n_threshold)]
    indices_to_remove = list(groups_to_remove.drop(columns='n').itertuples(index=False, name=None))

    grupos_indices = grupos.indices
    for idx in indices_to_remove:
        del grupos_indices[idx]
    
    grupos_reducidos = {}
    
    for grupo_key, grupo_idxs in grupos_indices.items():
        grupos_reducidos.update({grupo_key: df.iloc[grupo_idxs]})
    
    return groups_final_ordenado, grupos_reducidos

def get_barplot(df, titulo, x, y):
    '''
    Función que construye un gráfico de barras.

    Parámetros:
        df: dataset a graficar
        titulo: título del gráfico
        x: atributo del dataset, del eje x 
        y: atributo del dataset, del eje y
    '''      
    plt.figure(figsize=(30,25))
    sns.set(font_scale=1.1)
    sns.barplot(x=df.value_counts().head(30).index,y=df.value_counts().head(30),palette="viridis")

    plt.legend(title=titulo, fontsize=20)
    plt.xlabel(x, fontsize=11);
    plt.ylabel(y, fontsize=20)
    plt.title(titulo, fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=14)

def get_pie(df, titulo, dim):
    '''
    Función que construye un gráfico de "pastel".

    Parámetros:
        df: dataset a graficar
        titulo: título del gráfico
        dim: dimensiones del gráfico
    '''      
    Makers=df.value_counts().index
    df.value_counts().plot.pie(figsize=(dim, dim),autopct='%1.1f%%',labels=Makers,labeldistance=1.05,fontsize=dim,legend=False)
    plt.title(titulo, fontsize=30)    

def get_barplot_horiz(df, y, tit_y, x, tit_x):
    '''
    Función que construye un gráfico de barras horizontal.

    Parámetros:
        df: dataset a graficar 
        y: atributo del dataset, del eje y
        tit_y: titulo del eje y
        x: atributo del dataset, del eje x
        tit_x: titulo del eje x
    '''      
    plt.figure( figsize = (30,25))
    sns.barplot(y=y,x=x,data=df)

    plt.ylabel(tit_y,fontsize=20)
    plt.xlabel(tit_x,fontsize=20)

def get_histplot(df, x, limite):
    '''
    Función que grafica un histograma.

    Parámetros:
        df: dataset a graficar
        x: atributo del dataset
        limite: limite del eje x
    '''
    plt.figure( figsize = (20,20))
    g=sns.histplot(x=x,data=df,kde=True)
    plt.title('Histograma de ' + str(x), fontsize=20)
    g=g.set(xlim=(0,limite))
    plt.show(g)    

def get_scatterplot(df, x, y):
    '''
    Función que grafica un scatterplot.

    Parámetros:
        df: dataset a graficar
        x: atributo del eje x
        y: atributo del eje y
    '''    
    plt.figure(figsize=(25,16))
    sns.scatterplot(x=x,y=y,data=df,color='purple',alpha=0.3)
    plt.ylabel(y,fontsize=20)
    plt.xlabel(x,fontsize=20)

def get_scatterplot_hue(df, x, y, hue):
    '''
    Función que grafica un scatterplot, con legenda .

    Parámetros:
        df: dataset a graficar
        x: atributo del eje x
        y: atributo del eje y
        hue: atributo de agrupación que producirá puntos con diferentes colores (legenda)
    '''        
    plt.figure(figsize=(25,16))
    ax=sns.scatterplot(x=x,y=y,data=df,color='purple',alpha=0.25,hue=hue)
    plt.ylabel(y,fontsize=20)
    plt.xlabel(x,fontsize=20)    
    plt.setp(ax.get_legend().get_texts(), fontsize='22') # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='32') # for legend title

def get_boxplot(df, x, y):
    '''
    Función que grafica un boxplot.

    Parámetros:
        df: dataset a graficar
        x: atributo del eje x
        y: atributo del eje y
    ''' 
    plt.figure(figsize=(25,10))
    sorted_nb = df.groupby([x])[y].median().sort_values()
    sns.boxplot(x=df[x], y=df[y], order=list(sorted_nb.index))
    plt.xticks(rotation=70)

def get_heatmap(df):
    '''
    Función que mapea con colores las correlaciones ("mapa de calor").

    Parámetros:
        df: dataset a graficar
    '''
    plt.figure(figsize=(25,10))
    sns.set(font_scale=1.4)
    sns.heatmap(df.corr(),annot=True, annot_kws={"size": 16})

def regresion_train(grupo_df, cols_to_drop, model, var_to_estimate='precio'):
    '''
    Función que realiza la transformación, imputación y escalamiento de los datos, y luego
    el entrenamiento de estos.

    Parámetros:
        grupo_df
        cols_to_drop
        model
        var_to_estimate

    Retorno:
        minipipe (Pipeline)
    '''      
    grupo_para_regresion = grupo_df.drop(columns=cols_to_drop)
    oe_col = list(grupo_para_regresion.select_dtypes(include=['object']).columns)

    minipipe = Pipeline(steps=[
        ('enc', SklearnTransformerWrapper(transformer=OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan), variables=oe_col)),
        # esto es pq puede ser que en el test aparezcan nuevos modelos,marcas u otra variable categórica que no apareció en el train. En ese caso se asume como nan y luego se imputa.
        ('imp', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
        ('stc', StandardScaler()),
        ('model', model)
    ])

    X = grupo_para_regresion.drop(columns=var_to_estimate)
    y = grupo_para_regresion[var_to_estimate]
    minipipe.fit(X,y)
    return minipipe

def variable_de_grupo_en_dataframe(df, grupos_reducidos, variables_a_agrupar):
    '''
    Función que agrega una columna al dataframe para indicar a qué grupo pertenece cada observacion. 

    Parámetros:
        df 
        grupos_reducidos
        variables_a_agrupar

    Retorno:
        df
        indice_grupos
    '''      
    indice_grupos = {}
    count = 1
    df['grupo'] = pd.Series()
    for grupo_key, grupo in tqdm(grupos_reducidos.items()):
        codigo_grupo = f'g{count}'
        indice_grupos[codigo_grupo] = grupo_key
        variables_to_query = zip(variables_a_agrupar, list(grupo_key))
        query_text = ' & '.join([f'{t[0]} == "{t[1]}"' if type(
            t[1]) == str else f'{t[0]} == {t[1]}' for t in variables_to_query])
        indexes = df.query(query_text).index
        df.loc[indexes, 'grupo'] = codigo_grupo
        count += 1
    return df, indice_grupos

def entrenamiento_segmentado(indice_grupos, df_copy_train, vars_to_remove, var_to_estimate, model):
    '''
    Función que entrena un modelo para cada grupo incluído en indice_grupos. Previamente se eliminan los outliers por segmento.

    Parámetros:
        indice_grupos: diccionario con la codificación de cada grupo. Solo grupos con un N > N_minimo
        df_copy_train: dataframe de los datos que contiene una columna adicional indicando el grupo al que corresponde la muestra
        vars_to_remove: Variables que se remueven del dataframe para hacer el entrenamiento. 
                        Ej: ciudad, grupo, modelo, etc. Estas se sacan porque se ocuparon para hacer la segmentación. 
                        Como en cada segmento su valor será igual, no tiene sentido incluirlas en el modelo.
        var_to_estimate: Variable a predecir, típicamente `precio`
        model: Base model a usar como regresor

    Retorno:
        model_per_group: diccionario con los modelos indexados por grupo
    '''          
    model_per_group = {}
    print('Entrenando los modelos de cada segmento')
    print(f'Antes de sacar outliers N={len(df_copy_train)}')
    Nout=0
    for idx_grupo, tupla_grupo in tqdm(indice_grupos.items()):
        grupo = df_copy_train[df_copy_train.grupo == idx_grupo]
        if idx_grupo == 'g18':
            grupo.precio.hist(bins=20)
        outliter_grupo = IQR_method(grupo, 0, ['precio'])
        Nout = Nout+len(outliter_grupo)
        grupo = grupo.drop(outliter_grupo, axis=0).reset_index(drop=True)  # remoción de outliers
        if idx_grupo == 'g18':
            grupo.precio.hist(bins=20)
        model_per_group[idx_grupo] = regresion_train(grupo, vars_to_remove, model, var_to_estimate=var_to_estimate)
    #print(f'Después de sacar outliers N={len(df_copy_train)-Nout}')
    return model_per_group

def new_cars_estimations(model_per_group, df_test, indice_grupos, variables_a_agrupar, cols_to_drop, estimate='precio', roi_threshold=8):
    '''
    Función que realiza la predicción de precio y toma de decisión por muestra. 

    Parámetros:
        model_per_group:
        df_test:
        indice_grupos:
        variables_a_agrupar:
        cols_to_drop:
        estimate:
        roi_threshold:

    Retorno:
        info_dict: resultados
    '''      
    grupos_test_list = []
    grupos_test = df_test.groupby(by=variables_a_agrupar)
    list_of_tuples_str = [','.join(str(subel) for subel in el) for el in list(indice_grupos.values())]
    print('Encontrando los grupos para las muestras de test...')
    for grupo_key, grupo in tqdm(grupos_test):
        tuple_to_find_str = ','.join(str(subel) for subel in grupo_key)
        try:
            index_in_train_of_test = list_of_tuples_str.index(tuple_to_find_str)
        except ValueError as vr:
            # print(f'La tupla {grupo_key} no se encontró en la data de train. No es posible tomar decisiones respecto a esa muestra.')
            continue
        indice_grupo = list(indice_grupos.keys())[index_in_train_of_test]
        grupos_test_list.append((grupo_key, indice_grupo, model_per_group[indice_grupo]))

    info_dict = {}
    grupo_para_regresion = df_test.drop(columns=cols_to_drop)

    print('Prediciendo precio de mercado y tomando decisión de compra para las muestras de test...')
    for tupla_grupo, idx_grupo, modelo in tqdm(grupos_test_list):
        
        variables_to_query = zip(variables_a_agrupar, list(tupla_grupo))
        query_text = ' & '.join([f'{t[0]} == "{t[1]}"' if type(t[1]) == str else f'{t[0]} == {t[1]}' for t in variables_to_query])
        df_test_group = df_test.query(query_text).drop(columns=cols_to_drop)
        
        if df_test_group.shape[0] <= 0:
            continue
        # print(tupla_grupo)
        # print(df_test_group.columns)
        X_test = df_test_group.drop(columns=estimate)
        # print(X_test.columns)
        y_test = df_test_group[estimate]
        y_pred = modelo.predict(X_test)
        
        results = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred, 'index': y_test.index})
        results['roi'] = (100/results.y_test) * (results.y_pred - results.y_test) # según nuestra investigación el roi debería ser mayor al menos al 8% para tener buenas ganancias
        results['decision'] = 1*(results['roi'] > roi_threshold)
        
        casos_compra = results[results['decision'] == True].sort_values(by='roi', ascending=False).index
        casos_totales = results.sort_values(by='roi', ascending=False).index
        casos_no_comprados = casos_totales.difference(casos_compra)
        compras = df_test.loc[casos_compra]
        no_compras = df_test.loc[casos_no_comprados]
        
        paquete_info = {
            'grupo': (idx_grupo, tupla_grupo),
            'casos_de_compra': compras,
            'casos_de_no_compra': no_compras,
            'totales_analizados': df_test,
            'resultados_precios': results
        }
        
        info_dict[idx_grupo] = paquete_info
    return info_dict

def autos_comprados_results_plot(results):
    _, axs = plt.subplots(1, 2)
    results.plot.scatter(ax=axs[0], x='y_pred', y='y_test', c='roi', cmap="RdBu")
    axs[1] = results.plot.scatter(
        ax=axs[1], x='y_pred', y='y_test', c='decision', cmap="RdBu")
    axs[1] = add_identity(axs[1])
    axs[1].set_aspect(1)
    plt.show()

def add_identity(axes, *line_args, **line_kwargs):
    identity, = axes.plot([], [], *line_args, **line_kwargs, color='green')

    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes

def IQR_method(df, n, features, outlier_step=1.5):
    """
    Takes a dataframe and returns an index list corresponding to the observations 
    containing more than n outliers according to the Tukey IQR method.
    """
    outlier_list = []

    for column in features:

        # 1st quartile (25%)
        # print(df[column])
        Q1 = np.percentile(df[column], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[column], 75)

        # Interquartile range (IQR)
        IQR = Q3 - Q1

        # outlier step
        outlier_step = 2.5 * IQR

        # Determining a list of indices of outliers
        outlier_list_column = df[(
            df[column] < Q1 - outlier_step) | (df[column] > Q3 + outlier_step)].index

        # appending the list of outliers
        outlier_list.extend(outlier_list_column)

    # selecting observations containing more than x outliers
    outlier_list = Counter(outlier_list)
    multiple_outliers = list(k for k, v in outlier_list.items() if v > n)

    # Calculate the number of records below and above lower and above bound value respectively
    df1 = df[df[column] < Q1 - outlier_step]
    df2 = df[df[column] > Q3 + outlier_step]

    # print('Total number of outliers is:', df1.shape[0]+df2.shape[0])

    return multiple_outliers

def StDev_method(df, n, features):
    """
    Takes a dataframe df of features and returns an index list corresponding to the observations 
    containing more than n outliers according to the standard deviation method.
    """
    outlier_indices = []

    for column in features:
        # calculate the mean and standard deviation of the data frame
        data_mean = df[column].mean()
        data_std = df[column].std()

        # calculate the cutoff value
        cut_off = data_std * 3

        # Determining a list of indices of outliers for feature column
        outlier_list_column = df[(
            df[column] < data_mean - cut_off) | (df[column] > data_mean + cut_off)].index

        # appending the found outlier indices for column to the list of outlier indices
        outlier_indices.extend(outlier_list_column)

    # selecting observations containing more than x outliers
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)

    # Calculate the number of records below and above lower and above bound value respectively
    df1 = df[df[column] > data_mean + cut_off]
    df2 = df[df[column] < data_mean - cut_off]
    print('Total number of outliers is:', df1.shape[0] + df2.shape[0])

    return multiple_outliers

def z_score_method(df, n, features):
    """
    Takes a dataframe df of features and returns an index list corresponding to the observations 
    containing more than n outliers according to the z-score method.
    """
    outlier_list = []

    for column in features:
        # calculate the mean and standard deviation of the data frame
        data_mean = df[column].mean()
        data_std = df[column].std()
        threshold = 3

        z_score = abs((df[column] - data_mean)/data_std)

        # Determining a list of indices of outliers for feature column
        outlier_list_column = df[z_score > threshold].index

        # appending the found outlier indices for column to the list of outlier indices
        outlier_list.extend(outlier_list_column)

    # selecting observations containing more than x outliers
    outlier_list = Counter(outlier_list)
    multiple_outliers = list(k for k, v in outlier_list.items() if v > n)

    # Calculate the number of outlier records
    df1 = df[z_score > threshold]
    print('Total number of outliers is:', df1.shape[0])

    return multiple_outliers

def z_scoremod_method(df, n, features):
    """
    Takes a dataframe df of features and returns an index list corresponding to the observations 
    containing more than n outliers according to the z-score modified method.
    """
    outlier_list = []

    for column in features:
        # calculate the mean and standard deviation of the data frame
        data_mean = df[column].mean()
        data_std = df[column].std()
        threshold = 3
        MAD = median_abs_deviation

        mod_z_score = abs(0.6745*(df[column] - data_mean)/MAD(df[column]))

        # Determining a list of indices of outliers for feature column
        outlier_list_column = df[mod_z_score > threshold].index

        # appending the found outlier indices for column to the list of outlier indices
        outlier_list.extend(outlier_list_column)

    # selecting observations containing more than x outliers
    outlier_list = Counter(outlier_list)
    multiple_outliers = list(k for k, v in outlier_list.items() if v > n)

    # Calculate the number of outlier records
    df1 = df[mod_z_score > threshold]
    print('Total number of outliers is:', df1.shape[0])

    return multiple_outliers


def scatter_compras_roi(results):
    fig, axs = plt.subplots(1, 2)

    axs[0] = results.plot.scatter(
        ax=axs[0], x='y_pred', y='y_test', c='roi', cmap="RdBu")
    axs[1] = results.plot.scatter(
        ax=axs[1], x='y_pred', y='y_test', c='decision', cmap="RdBu")
    axs[0] = add_identity(axs[0])
    axs[1] = add_identity(axs[1])

    min_x = min(results.y_test.min(), results.y_pred.min())
    max_x = min(results.y_test.max(), results.y_pred.max())
    dif_x = max_x - min_x
    min_x = min_x - dif_x * 0.5
    max_x = max_x + dif_x * 0.5

    axs[0].set_xlim([min_x, max_x])
    axs[0].set_ylim([min_x, max_x])
    axs[1].set_xlim([min_x, max_x])
    axs[1].set_ylim([min_x, max_x])

    axs[0].set_aspect(1)
    axs[1].set_aspect(1)
    # plt.draw()
    plt.show()

######## VARIABLES #########

lista_cambios = {
    # https://en.wikipedia.org/wiki/List_of_Acura_vehicles
    'acura': [('ilx.*', 'ilx'), ('tsx.*', 'tsx'), ('nsx.*', 'nsx'), ('rlx.*', 'rlx'), ('rdx.*', 'rdx'), ('zdx.*', 'zdx'), ('tlx.*', 'tlx'),('mdx.*', 'mdx'), ('integra.*', 'integra'), ('tl[^x].*', 'tl'), ('rsx.*', 'rsx'), ('cl.*', 'cl'), ('rl[^xs].*', 'rl'), ('slx.*', 'slx')],
    # https://en.wikipedia.org/wiki/List_of_Audi_vehicles
    'audi': [('^s8.*', 's8'), ('^s7.*', 's7'), ('^s6.*', 's6'), ('^s5.*', 's5'), ('^s4.*', 's4'), ('^s3.*', 's3'), ('^a8.*', 'a8'), ('^a7.*', 'a7'), ('^a6.*', 'a6'), ('^a5.*', 'a5'), ('^a4.*', 'a4'), ('^a3.*', 'a3'), ('^q5.*', 'q5'), ('^q4.*', 'q4'), ('^q3.*', 'q3'), ('^q6.*', 'q6'), ('^q7.*', 'q7'), ('^tt.*', 'tt'), ('^r8.*', 'r8'), ('^allroad.*', 'allroad'), ('^rs4.*', 'rs4'), ('^rs6.*', 'rs6'), ('^sq5.*', 'sq5')],
    # https://en.wikipedia.org/wiki/List_of_BMW_vehicles, https://www.bmw.cl/es/all-models.html
    'bmw': [('^m2.*', 'm2'), ('^m3.*', 'm3'), ('^m4.*', 'm4'), ('^m5.*', 'm5'), ('^m6.*', 'm6'), ('^x1.*', 'x1'), ('^x2.*', 'x2'), ('^x3.*', 'x3'), ('^x4.*', 'x4'), ('^x5.*', 'x5'), ('^x6.*', 'x6'), ('^z3.*', 'z3'), ('^z4.*', 'z4'), ('^i3.*', 'i3')],
    # https://en.wikipedia.org/wiki/List_of_Buick_vehicles, https://es.wikipedia.org/wiki/Buick
    'buick': [('^cascada.*', 'cascada'), ('^century.*', 'century'), ('^enclave.*', 'enclave'), ('^encore.*', 'encore'), ('^envision.*', 'envision'), ('^lacrosse.*', 'lacrosse'), ('^lesabre.*', 'lesabre'), ('^rainier.*', 'rainier'), ('^regal.*', 'regal'), ('^rendezvous.*', 'rendezvous'), ('^verano.*', 'verano')],
    # https://en.wikipedia.org/wiki/Bentley#List_of_Bentley_vehicles
    'bentley': [('^continental.*', 'continental'), ('^mulsanne.*', 'mulsanne')],
    # https://en.wikipedia.org/wiki/List_of_Cadillac_vehicles
    'cadillac': [('^ats.*', 'ats'), ('^ct.*', 'ct'), ('^deville.*', 'deville'), ('^escalade.*', 'escalade'), ('^srx.*', 'srx'), ('^sts.*', 'sts'), ('^xlr.*', 'xlr'), ('^xt5.*', 'xt5'), ('^xts.*', 'xts')],
    # https://en.wikipedia.org/wiki/List_of_Chevrolet_vehicles
    'chevrolet': [('^avalanche.*', 'avalanche'), ('^aveo.*', 'aveo'), ('^blazer.*', 'blazer'), ('^camaro.*', 'camaro'), ('^cavalier.*', 'cavalier'), ('^cobalt.*', 'cobalt'), ('^colorado.*', 'colorado'), ('^corvette.*', 'corvette'), ('^cruze.*', 'cruze'), ('^equinox.*', 'equinox'), ('^hhr.*', 'hhr'), ('^impala.*', 'impala'), ('^malibu.*', 'malibu'), ('^sonic.*', 'sonic'), ('^spark.*', 'spark'), ('^suburban.*', 'suburban'), ('^tahoe.*', 'tahoe'), ('^tracker.*', 'tracker'), ('^trailblazer.*', 'trailblazer'), ('^traverse.*', 'traverse'), ('^trax.*', 'trax'), ('^uplander.*', 'uplander'), ('^venture.*', 'venture'), ('^volt.*', 'volt')],
    # https://en.wikipedia.org/wiki/List_of_Chrysler_vehicles
    'chrysler': [('^aspen.*', 'aspen'), ('^pacifica.*', 'pacifica'), ('^sebring.*', 'sebring'), ('^200.*', '200'), ('^300.*', '300')],
    # https://en.wikipedia.org/wiki/List_of_Dodge_vehicles
    'dodge': [('^avenger.*', 'avenger'), ('^caliber.*', 'caliber'), ('^caravan.*', 'caravan'), ('^challenger.*', 'challenger'), ('^charger.*', 'charger'), ('^dakota.*', 'dakota'), ('^dart.*', 'dart'), ('^durango.*', 'durango'), ('^journey.*', 'journey'), ('^neon.*', 'neon'), ('^nitro.*', 'nitro'), ('^sprinter.*', 'sprinter'), ('^stratus.*', 'stratus'), ('^viper.*', 'viper')],
    # https://en.wikipedia.org/w/index.php?go=Go&search=ferrari+models&title=Special%3ASearch&ns0=1
    'ferrari': [('^california.*', 'california'), ('^f12.*', 'f12'), ('^360.*', '360'), ('^430.*', '430'), ('^ff.*', 'ff')],
    # Ford
    'ford': [('^edge.*', 'edge'), ('^escape.*', 'escape'), ('^escort.*', 'escort'), ('^excursion.*', 'excursion'), ('^expedition.*', 'expedition'), ('^explorer.*', 'explorer'), ('^f150.*', 'f150'), ('^f250.*', 'f250'), ('^fiesta.*', 'fiesta'), ('^flex.*', 'flex'), ('^focus.*', 'focus'), ('^fusion.*', 'fusion'), ('^mustang.*', 'mustang'), ('^ranger.*', 'ranger'), ('^taurus.*', 'taurus')],
    # GMC
    'gmc': [('^acadia.*', 'acadia'), ('^envoy.*', 'envoy'), ('^canyon.*', 'canyon'), ('^jimmy.*', 'jimmy'), ('^sonoma.*', 'sonoma'), ('^suburban.*', 'suburban'), ('^terrain.*', 'terrain'), ('^yukon.*', 'yukon')],
    # Honda
    'honda': [('^civic.*', 'civic'), ('^cr-.*', 'cr-'), ('^fit.*', 'fit'), ('^hr-.*', 'hr-'), ('^insight.*', 'insight'), ('^odyssey.*', 'odyssey'), ('^passport.*', 'passport'), ('^pilot.*', 'pilot'), ('^ridgeline.*', 'ridgeline')],
    # HUMMER
    'hummer': [('^h2.*', 'h2'), ('^h3.*', 'h3')],
    # Hyundai
    'hyundai': [('^accent.*', 'accent'), ('^azera.*', 'azera'), ('^elantra.*', 'elantra'), ('^equus.*', 'equus'), ('^genesis.*', 'genesis'), ('^sonata.*', 'sonata'), ('^tucson.*', 'tucson'), ('^veloster.*', 'veloster'), ('^veracruz.*', 'veracruz')],
    # INFINITI
    'infiniti': [('^ex3.*', 'ex3'), ('^fx3.*', 'fx3'), ('^i30.*', 'i30'), ('^m35.*', 'm35'), ('^m56.*', 'm56'), ('^q40.*', 'q40'), ('^q45.*', 'q45') , ('^q50.*', 'q50'), ('^q60.*', 'q60'), ('^q70.*', 'q70'), ('^qx30.*', 'qx30'), ('^qx50.*', 'qx50'), ('^qx56.*', 'qx56'), ('^qx60.*', 'qx60'), ('^qx70.*', 'qx70'), ('^qx80.*', 'qx80')],
    # Isuzu
    'isuzu': [('^ascender.*', 'ascender'), ('^hombre.*', 'hombre')],
    # Jaguar
    'jaguar': [('^f-pace.*', 'f-pace'), ('^f-type.*', 'f-type'), ('^xe.*', 'xe'), ('^xf.*', 'xf'), ('^xj.*', 'xj'), ('^xk.*', 'xk')],
    # Jeep
    'jeep': [('^cherokee.*', 'cherokee'), ('^commander.*', 'commander'), ('^compass.*', 'compass'), ('^liberty.*', 'liberty'), ('^patriot.*', 'patriot'), ('^renegade.*', 'renegade'), ('^wrangler.*', 'wrangler')],
    # Kia
    'kia': [('^borrego.*', 'borrego'), ('^cadenza.*', 'cadenza'), ('^forte.*', 'forte'), ('^k900.*', 'k900'), ('^niro.*', 'niro'), ('^optima.*', 'optima'), ('^rio.*', 'rio'), ('^sedona.*', 'sedona'), ('^sorento.*', 'sorento'), ('^spectra.*', 'spectra'), ('^soul.*', 'soul'), ('^sportage.*', 'sportage')],
    # Lamborghini
    'Lamborghini': [('^aventador.*', 'aventador'), ('^gallardo.*', 'gallardo'), ('^huracan.*', 'huracan'), ('^murcielago.*', 'murcielago')],
    # Lexus
    'lexus': [('^ct.*', 'ct'), ('^es.*', 'es'), ('^gs.*', 'gs'), ('^gx.*', 'gx'), ('^nx.*', 'nx'), ('^rx.*', 'rx'), ('^rc.*', 'rc')],
    # Lincoln
    'lincoln': [('^aviator.*', 'aviator'), ('^continental.*', 'continental'), ('^mk.*', 'mk'), ('^navigator.*', 'navigator')],
    # Lotus
    'lotus': [('^evora.*', 'evora')],
    # Maserati
    'maserati': [('^ghibli.*', 'ghibli'), ('^gran.*', 'gran'), ('^quattroporte.*', 'quattroporte')],
    # Mazda
    'mazda': [('^cx.*', 'cx'), ('^mazda2.*', 'mazda2'), ('^mazda3.*', 'mazda3'), ('^mazda5.*', 'mazda5'), ('^mazda6.*', 'mazda6'), ('^tribute.*', 'tribute')],
    # McLaren
    'mclaren': [('^570.*', '570')],
    # Mercedes-Benz
    'mercedes-benz': [('^b-class.*', 'b-class'), ('^c-class.*', 'c-class'), ('^cla.*', 'cla'), ('^cl.*', 'cl'), ('^clk.*', 'clk'), ('^cls.*', 'cls'), ('^e-class.*', 'e-class'), ('^g-class.*', 'g-class'), ('^gla.*', 'gla'), ('^gl.*', 'gl'), ('^glegle.*', 'glegle'), ('^glk.*', 'glk'), ('^m-class.*', 'm-class'), ('^r-class.*', 'r-class'), ('^s-class.*', 's-class'), ('^slk-class.*', 'slk-class'), ('^sl-class.*', 'sl-class')],
    # Mercury
    'mercury': [('^cougar.*', 'cougar'), ('^mariner.*', 'mariner'), ('^mountaineer.*', 'mountaineer'), ('^village.*', 'village')],
    # Mitsubishi
    'mitsubishi': [('^eclipse.*', 'eclipse'), ('^endeavor.*', 'endeavor'), ('^galant.*', 'galant'), ('^lancer.*', 'lancer'), ('^mirage.*', 'mirage'), ('^outlander.*', 'outlander'), ('^raider.*', 'raider')],
    # Nissan
    'nissan': [('^altima.*', 'altima'), ('^370.*', '370'), ('^armada.*', 'armada'), ('^cube.*', 'cube'), ('^frontier.*', 'frontier'), ('^gt-.*', 'gt-'), ('^juke.*', 'juke'), ('^lea.*', 'lea'), ('^maxima.*', 'maxima'), ('^murano.*', 'murano'), ('^nv.*', 'nv'), ('^pathfinder.*', 'pathfinder'), ('^quest.*', 'quest'), ('^rogue.*', 'rogue'), ('^sentra.*', 'sentra'), ('^titan.*', 'titan'), ('^versa.*', 'versa'), ('^xterra.*', 'xterra')],
    # Oldsmobile
    'oldsmobile': [('^alero.*', 'alero'), ('^cutlass.*', 'cutlass')],
    # Plymouth
    'plymouth': [('^voyage.*', 'voyage'), ('^370.*', '370'), ('^armada.*', 'armada'), ('^cube.*', 'cube'), ('^frontier.*', 'frontier'), ('^gt-.*', 'gt-'), ('^juke.*', 'juke'), ('^lea.*', 'lea'), ('^maxima.*', 'maxima'), ('^murano.*', 'murano'), ('^nv.*', 'nv'), ('^pathfinder.*', 'pathfinder'), ('^quest.*', 'quest'), ('^rogue.*', 'rogue'), ('^sentra.*', 'sentra'), ('^titan.*', 'titan'), ('^versa.*', 'versa'), ('^xterra.*', 'xterra')],
    # Pontiac
    'pontiac': [('^montana.*', 'montana'), ('^sunfire.*', 'sunfire'), ('^torrent.*', 'torrent')],
    # Porsche
    'porsche': [('^911.*', '911'), ('^boxster.*', 'boxster'), ('^cayenne.*', 'cayenne'), ('^cayman.*', 'cayman'), ('^macan.*', 'macan'), ('^panamera.*', 'panamera')],
    # Ram
    'ram': [('^1500.*', '1500'), ('^2500.*', '2500'), ('^3500.*', '3500')],
    # Rolls-Royce
    'rolls-royce': [('^dawn.*', 'dawn'), ('^ghost.*', 'ghost'), ('^phantom.*', 'phantom')],
    # Saturn
    'saturn': [('^astra.*', 'astra'), ('^lsl.*', 'lsl'), ('^lwlw.*', 'lwlw'), ('^outlook.*', 'outlook'), ('^slsl.*', 'slsl'), ('^swsw.*', 'swsw'), ('^vue.*', 'vue')],
    # Subaru
    'subaru': [('^brz.*', 'brz'), ('^impreza.*', 'impreza'), ('^legacy.*', 'legacy'), ('^crosstrek.*', 'crosstrek'), ('^outback.*', 'outback'), ('^tribeca.*', 'tribeca'), ('^forester.*', 'forester'), ('^wrx.*', 'wrx')],
    # Suzuki
    'suzuki': [('^sx4.*', 'sx4'), ('^kizashi.*', 'kizashi'), ('^equator.*', 'equator'), ('^xl.*', 'xl'), ('^vitara.*', 'vitara'), ('^sidekick.*', 'sidekick')],
    # Tesla
    'tesla': [('^roadster.*', 'roadster')],
    # Toyota
    'toyota': [('^camry.*', 'camry'), ('^yaris.*', 'yaris'), ('^prius.*', 'prius'), ('^corolla.*', 'corolla'), ('^matrix.*', 'matrix'), ('^highlander.*', 'highlander'), ('^rav4.*', 'rav4'), ('^avalon.*', 'avalon'), ('^86.*', '86'), ('^venzale.*', 'venzale'), ('^echo.*', 'echo'), ('^tundra.*', 'tundra'), ('^sienna.*', 'sienna'), ('^sequoia.*', 'sequoia'), ('^celica.*', 'celica'), ('^4runner.*', '4runner'), ('^tacoma.*', 'tacoma')],
    # Volkswagen
    'volkswagen': [('^passat.*', 'passat'), ('^golf.*', 'golf'), ('^jetta.*', 'jetta'), ('^cc.*', 'cc'), ('^routan.*', 'routan'), ('^e-golf.*', 'e-golf'), ('^eos.*', 'eos'), ('^gli.*', 'gli'), ('^rabbit.*', 'rabbit'), ('^beetle.*', 'beetle'), ('^tiguan.*', 'tiguan'), ('^touareg.*', 'touareg'), ('^gti.*', 'gti'), ('^cabrio.*', 'cabrio')],
    # Volvo
    'volvo': [('^c30.*', 'c30'), ('^c70.*', 'c70'), ('^s40.*', 's40'), ('^s60.*', 's60'), ('^s70.*', 's70'), ('^s80.*', 's80'), ('^s90.*', 's90'), ('^v40.*', 'v40'), ('^v50.*', 'v50'), ('^v60.*', 'v60'), ('^v702.*', 'v702'), ('^xc60.*', 'xc60'), ('^xc70.*', 'xc70'), ('^xc90.*', 'xc90')]
}
# Consideramos que no aplica a los modelos: Alfa, AM, Aston, Fiat, Fisker, Freightliner, Genesis, Geo, Land, MINI, Saab, Scion, smart

# Segmentación de marcas de vehículos en Gama Alta, media y Baja

gamas={
'lujo':['mcLaren', 'rolls-royce', 'ferrari', 'lamborghini', 'bentley', 'aston', 'maybach'],
'alta': ['tesla', 'am', 'ram', 'porsche',
'alfa', 'lotus', 'maserati', 'fisker', 'genesis', 'land', 'jaguar', 'mercedes-benz',
'audi', 'freightliner', 'gmc', 'cadillac', 'bmw', 'lexus', 'infiniti', 'volvo', 'acura', 'lincoln'],
'media': ['jeep', 'chevrolet', 'ford', 'plymouth', 'toyota', 'subaru', 'hummer', 'buick', 'dodge',
'honda', 'nissan', 'mini', 'chrysler', 'mazda', 'kia', 'volkswagen', 'hyundai', 'mitsubishi', 
'scion', 'fiat', 'smart', 'pontiac', 'mercury', 'suzuki', 'saab', 'saturn', 'geo', 'isuzu', 'oldsmobile']
}

# Años de vehículos
anio_vehiculo={'V': 1997, 'W':1998, 'X':1999, 'Y':2000, '1':2001, '2':2002, '3':2003, '4':2004, '5':2005, '6':2006, '7':2007, '8':2008, '9':2009, 'A':2010, 'B':2011, 'C':2012, 'D':2013, 'E':2014, 'F':2015, 'G':2016, 'H':2017, 'J':2018}
