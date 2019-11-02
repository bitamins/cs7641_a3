# Data Tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Clustering
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# Dimensionality Reduction
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Neural Network
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve

# Metrics, for validation of clustering choice of k
from sklearn.metrics import homogeneity_score, v_measure_score, completeness_score, mutual_info_score, adjusted_mutual_info_score, calinski_harabasz_score ,accuracy_score,pairwise_distances,accuracy_score

# Model Tools
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# yellowbrick
from yellowbrick.cluster import KElbowVisualizer

# STD Lib
import time
import logging

LOGGER = None

def create_logger():
    # only needs to be called once
    logger_active = True
    logger_name = __name__
    format_string = '%(asctime)s - %(levelname)s - ln:%(lineno)d\t| %(message)s'
        
    formatter = logging.Formatter(format_string)
     
    file_handler = logging.FileHandler(logger_name+'.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    logger.disabled = not logger_active
    return logger
    

def get_Xy(df,labels):
    df_X = df.drop([labels],axis=1)
    df_y = df[[labels]]
    return df_X,df_y

def get_data_df(id='wage'):
    LOGGER.info('getting {} dataframe'.format(id))
    if id == 'wage':
        df = pd.read_csv('data/balanced_wage_cleaned.csv',index_col=0)
        return get_Xy(df,'wage-class')
    elif id == 'wine':
        df = pd.read_csv('data/balanced_wine_cleaned.csv',index_col=0)
        columns = ['Alcohol',
                    'Malic acid',
                    'Ash',
                    'Alcalinity of ash',
                    'Magnesium',
                    'Total phenols',
                    'Flavanoids',
                    'Nonflavanoid phenols',
                    'Proanthocyanins',
                    'Color intensity',
                    'Hue',
                    'OD280/OD315 of diluted wines',
                    'Proline',
                    'class']
        df.columns = columns
        return get_Xy(df,'class')
                  
    
def run_kmeans(X_train, X_test, y_train, y_test):
    LOGGER.info('kmeans, train: {}, test: {}'.format(X_train.shape[0],X_test.shape[0]))
    
    # max_clusters = 7
    # clusters=[2**x for x in range(1,max_clusters)]
    clusters=[x for x in range(1,100)]
    # split_ratio = 0.33
    # X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=split_ratio,random_state=0)
    # LOGGER.debug('train test split: {}'.format(split_ratio))
    
    model = KMeans(random_state=0)
    
    
    # kms = [KMeans(n_clusters=i) for i in clusters]
    # choose_scores = [kms[i].fit(X_train).score(X) for i in range(len(kms))]
    
    
    # Validation
    
    score_fns = [
            # mutual_info_score,
            v_measure_score,
            homogeneity_score,
            completeness_score,
            # adjusted_mutual_info_score,
            # calinski_harabasz_score,
        ]
    
    # validation_score = pd.DataFrame(index=clusters,columns=sum([[score.__name__+'_train',score.__name__+'_test'] for score in score_fns],[]))
    validation_score = pd.DataFrame(index=clusters)
    choose_score = pd.DataFrame(index=clusters,columns=['score'])
    
    for k in clusters:
        # model = KMeans(random_state=0)
        LOGGER.debug('clusters: {}'.format(k))
        model.set_params(n_clusters=k)
        model.fit(X_train)
        sse_score = model.score(X_train)
        choose_score.loc[k,'score'] = sse_score
                          
        for score in score_fns:
            LOGGER.debug('evaluation: {}'.format(score.__name__))
            # validation_score.loc[k,score.__name__+'_train'] = score(y_train[y_train.columns[0]],model.predict(X_train))
            validation_score.loc[k,score.__name__+'_test'] = score(y_test[y_test.columns[0]],model.predict(X_test))
            # validation_score.loc[k,'k'] = k
            
    return validation_score,choose_score,model

def run_gm(X_train, X_test, y_train, y_test):
    LOGGER.info('kmeans, train: {}, test: {}'.format(X_train.shape[0],X_test.shape[0]))
    
    # max_clusters = 7
    # clusters=[2**x for x in range(1,max_clusters)]
    clusters=[x for x in range(1,20)]
    
    # split_ratio = 0.33
    # X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=split_ratio,random_state=0)
    # LOGGER.debug('train test split: {}'.format(split_ratio))
    
    model = GaussianMixture(random_state=0)
    
    score_fns = [
            # mutual_info_score,
            v_measure_score,
            homogeneity_score,
            completeness_score,
            # adjusted_mutual_info_score,
            # calinski_harabasz_score,
        ]
    
    # df_scores = pd.DataFrame(index=clusters,columns=sum([[score.__name__+'_train',score.__name__+'_test'] for score in score_fns],[]))
    df_scores = pd.DataFrame(index=clusters)
    
    for k in clusters:
        # model = GaussianMixture(random_state=0)
        LOGGER.debug('clusters: {}'.format(k))
        model.set_params(n_components=k)
        model.fit(X_train)
        sse_score = model.score(X_train)
                          
        for score in score_fns:
            LOGGER.debug('evaluation: {}'.format(score.__name__))
            # df_scores.loc[k,score.__name__+'_train'] = score(y_train[y_train.columns[0]],model.predict(X_train))
            df_scores.loc[k,score.__name__+'_test'] = score(y_test[y_test.columns[0]],model.predict(X_test))
            # df_scores.loc[k,'k'] = k
            
    return df_scores,model
    
        
def plot_scores(df,title,x,y,filename='scores',logscale=False):
    LOGGER.info('plotting {}'.format(title))
    plt.clf()
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    if logscale: plt.yscale('log')

    plt.grid()

    for col in df.columns:
        plt.plot(df.reset_index()['index'],df[col],'o-',label=col,alpha=0.8)
        
    plt.legend(loc="best")
    plt.savefig('plots/'+'_'.join([title,x,y]))
    

def plot_eigen(df,title,x,y,filename='eigen'):
    LOGGER.info('plotting {}'.format(title))
    plt.clf()
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid()
    
    sig_vec = [np.abs(i)/np.sum(df['pca_eigen']) for i in df['pca_eigen']]
    plt.step(range(0,len(sig_vec)),np.cumsum(sig_vec),label='cumulative explained variance')
    
    
    plt.bar(range(0,len(sig_vec)),sig_vec,align='center',label='explained variance')
    # for col in df.columns:
    #     plt.plot(df.reset_index()['index'],df[col],'o-',label=col,alpha=0.8)
        
    plt.legend(loc='best')
    plt.savefig('plots/'+'_'.join([title,x,y]))
    
def plot_projection_loss(df,title,x,y,filename='rpl'):
    LOGGER.info('plotting {}'.format(title))
    plt.clf()
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid()
    
    # print(df)
        
    plt.bar(df.reset_index()['index'],df['MSE'],align='center')
        
    plt.xticks(rotation=45,ha='right')
    
    plt.tight_layout()
        
    plt.legend(loc='best')
    plt.savefig('plots/'+'_'.join([title,x,y]))
    
    
def run_pca(X,y,n_components):
    LOGGER.info('pca...')
    
    
    split_ratio = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=split_ratio,random_state=0)
    LOGGER.debug('train test split: {}'.format(split_ratio))
    
    model = PCA(random_state=0)
    X_train_pca = model.fit_transform(X_train)
    X_test_pca = model.transform(X_test)
    eigen_df = pd.DataFrame(data=model.explained_variance_,columns=['pca_eigen'],index=range(len(model.explained_variance_)))

    model_df = pd.DataFrame(model.components_,columns=X.columns,index=range(model.components_.shape[0]))
    # print(model_df)
    
    
    


    model = PCA(n_components=n_components,random_state=0)
    X_train_pca = model.fit_transform(X_train)
    X_test_pca = model.transform(X_test)
    
    X_projected = model.inverse_transform(X_train_pca)
    
    projection_loss = pd.DataFrame(data=((X_train - X_projected)**2).mean(),columns=['MSE'])
        
    
    # LOGGER.info('PCA: components {}, projection-loss {}'.format(n_components,projection_loss))
    
    
    # y_pred = model.score(X_test_pca,y_test)
    
    kmeans_df,choose_df,km_model = run_kmeans(X_train_pca, X_test_pca, y_train, y_test)
    
    gm_df,gm_model = run_gm(X_train_pca, X_test_pca, y_train, y_test)
    

    
    return eigen_df, kmeans_df, gm_df,model,km_model,gm_model,projection_loss


def plot_kurt(df,title,x,y,filename='kurt'):
    LOGGER.info('plotting {}'.format(title))
    plt.clf()
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid()
    
    plt.plot(df.reset_index()['index'],df,'o-',label='distribution',alpha=0.8)
        
    plt.legend(loc='best')
    plt.savefig('plots/'+'_'.join([title,x,y]))

def run_ica(X,y,n_components):
    LOGGER.info('ica...')
    
    
    split_ratio = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=split_ratio,random_state=0)
    LOGGER.debug('train test split: {}'.format(split_ratio))
    
    model = FastICA(random_state=0,max_iter=1000)
    X_train_ica = model.fit_transform(X_train)
    X_test_ica = model.transform(X_test)
    # eigen_df = pd.DataFrame(data=model.explained_variance_,columns=['ica_eigen'],index=range(len(model.explained_variance_)))
    
    kurt_df = pd.DataFrame(X_train_ica)
    kurt_df = kurt_df.kurt(axis=0)
    # kurt_df[X_train.shape[1]] = kurt_df.abs().mean()
    kurt_df
    # print(kurt_df)

    model_df = pd.DataFrame(model.components_,columns=X.columns,index=range(model.components_.shape[0]))
    # print(model_df)


    # model = FastICA(random_state=0,max_iter=1000)
    # X_train_ica = model.fit_transform(X_train)
    # X_test_ica = model.transform(X_test)
    
    kmeans_df,choose_df,km_model = run_kmeans(X_train_ica, X_test_ica, y_train, y_test)
    
    gm_df,gm_model = run_gm(X_train_ica, X_test_ica, y_train, y_test)
    
        

    
    return kurt_df, kmeans_df, gm_df,model,km_model,gm_model


def run_ica_2(X,dataset):
    model = FastICA(random_state=0)
    
    result_df = pd.DataFrame()
    
    k_max = X.shape[1]
    if k_max > 120: k_max = 120
    for i in range(2,k_max+1):
        model.set_params(n_components=i)
        ica_data = pd.DataFrame(model.fit_transform(X)).kurt(axis=0) #kurtosis of ica results
        result_df.loc[i,'mean_kurtosis'] = ica_data.abs().mean()
        
    plt.clf()
    plt.title('ICA_Mean_Kurtosis_Per_K')
    plt.xlabel('K')
    plt.ylabel('Mean')
    plt.grid()
    
    plt.bar(range(2,result_df.shape[0]+2),result_df['mean_kurtosis'],align='center',label='mean kurtosis')
    
    LOGGER.info('ica max kurtosis on {}: k={}'.format(dataset,result_df.idxmax(axis=0)['mean_kurtosis']))
    
    plt.savefig('plots/'+'ica_kurt_'+dataset+'.png')


def run_rp(X,y,n_components):
    LOGGER.info('rp...')
    
    split_ratio = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=split_ratio,random_state=0)
    LOGGER.debug('train test split: {}'.format(split_ratio))
    
    
    model = SparseRandomProjection(n_components=X_train.shape[1],random_state=0)
    X_train_rp = model.fit_transform(X_train)
    X_test_rp = model.transform(X_test)
    
    # print(X_train.shape,X_train_rp.shape)
    
    kmeans_df,choose_df,km_model = run_kmeans(X_train_rp, X_test_rp, y_train, y_test)
    
    gm_df,gm_model = run_gm(X_train_rp, X_test_rp, y_train, y_test)
    
    
    return kmeans_df,gm_df,model,km_model,gm_model

def reconstruction_error(model,X):
    mcomp = model.components_.todense()
    pinv = np.linalg.pinv(mcomp)
    
    
    left = (pinv@mcomp)
    right = (X.T)
    rc = (left@right).T
    
    # print(rc)
    # print(X)
    # print(X-rc)
    
    X.columns = rc.columns
                            
    errs = np.square(X.subtract(rc))
    means = np.nanmean(errs)
    
    # print(means)
    # print(means.shape)
    return means

def run_rp_2(X,dataset):
    model = SparseRandomProjection()
    
    result_df = pd.DataFrame()
    
    iterations = 10
    
    k_max = X.shape[1]
    if k_max > 120: k_max = 120
    for i in range(2,k_max+1):
        LOGGER.info('rp: k={}'.format(i))
        vals = []
        for c in range(iterations):
            model.set_params(n_components=i)
            rp_X = model.fit_transform(X)
            
            r_e = reconstruction_error(model,X)
            
            # drp = pairwise_distances(rp_X)
            # dx = pairwise_distances(X)
            # vals.append(np.corrcoef(drp.ravel(),dx.ravel())[0,1])
            vals.append(r_e)
        result_df.loc[i,'mean'] = np.mean(vals)
        result_df.loc[i,'std'] = np.std(vals)
        
    plt.clf()
    plt.title('RP_Reconstruction_Error_'+dataset)
    plt.xlabel('K')
    plt.ylabel('error')
    plt.grid()
    
    plt.plot(result_df.reset_index()['index'],result_df['mean'],label='mean',color='blue')
    plt.fill_between(result_df.reset_index()['index'],result_df['mean'],result_df['mean']+result_df['std'],color='green',alpha=0.3)
    plt.fill_between(result_df.reset_index()['index'],result_df['mean'],result_df['mean']-result_df['std'],color='green',alpha=0.3)
    plt.savefig('plots/'+'rp_distances_'+dataset+'.png')
            
            
        
    


def run_lda(X,y,n_components,name):
    LOGGER.info('lda...')
    
    split_ratio = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=split_ratio,random_state=0)
    LOGGER.debug('train test split: {}'.format(split_ratio))
    
    
    model = LinearDiscriminantAnalysis()
    X_train_lda = model.fit_transform(X_train,y_train)
    X_test_lda = model.transform(X_test)

    CV = 5
    N_JOBS = 10

    train_sizes, train_scores, test_scores = learning_curve(LinearDiscriminantAnalysis(), X, y, scoring='accuracy', cv=CV, n_jobs=N_JOBS, shuffle=True)
    plot_learning_curve(train_scores, test_scores, train_sizes,name)
    
    
    kmeans_df,choose_df,km_model = run_kmeans(X_train_lda, X_test_lda, y_train, y_test)
    
    gm_df,gm_model = run_gm(X_train_lda, X_test_lda, y_train, y_test)
    
    
    return kmeans_df,gm_df,model,km_model,gm_model


def run_lda_2(X_train, X_test, y_train, y_test,dataset):
        
    # model = LinearDiscriminantAnalysis(solver='eigen',n_components=y_train.groupby(y_train.columns[0]).count().shape[0])
    model = LinearDiscriminantAnalysis(n_components=y_train.groupby(y_train.columns[0]).count().shape[0]-1)

    score_df = pd.DataFrame()
    
    # k_max = X_train.shape[1]-1
    # if k_max > 120: k_max = 120
    
    k_max = y_train.groupby(y_train.columns[0]).count().shape[0]
    
    for i in range(1,k_max):
        LOGGER.info('lda: k={}'.format(i))
        model.set_params(n_components=i)
        # model = LinearDiscriminantAnalysis(n_components=i)
        # lda_X = model.fit_transform(X_train,y_train[y_train.columns[0]])
        model.fit(X_train,y_train[y_train.columns[0]])
        
        # lda_test_X = model.transform(X_test)
        # y_pred = model.predict(lda_test_X)
        # print(y_pred)
        
        
        score_df.loc[i,'test_score'] = model.score(X_test,y_test[y_test.columns[0]])
        score_df.loc[i,'train_score'] = model.score(X_train,y_train[y_train.columns[0]])
        
    print(score_df)
        
    model = LinearDiscriminantAnalysis(n_components=y_train.groupby(y_train.columns[0]).count().shape[0]-1)
    
    model.fit(X_train,y_train[y_train.columns[0]])
    
    result_df = pd.DataFrame(data=model.explained_variance_ratio_,columns=['ex_variance'],index=range(len(model.explained_variance_ratio_)))

        
        
    title = 'lda_explained_variance'
    x = 'components'
    y = 'variance contributed'
    LOGGER.info('plotting {}'.format(title))
    plt.clf()
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid()
    
    # sig_vec = [np.abs(i)/np.sum(result_df['ex_variance']) for i in result_df['ex_variance']]
    plt.step(range(0,result_df.shape[0]),np.cumsum(result_df['ex_variance']),label='cumulative explained variance')
    
    
    plt.bar(range(0,result_df.shape[0]),result_df['ex_variance'],align='center',label='explained variance')
    # for col in df.columns:
    #     plt.plot(df.reset_index()['index'],df[col],'o-',label=col,alpha=0.8)
        
    # plt.legend(loc='best')
    # plt.savefig('plots/'+'_'.join([title,x,y,dataset]))
        
    # plt.clf()
    # plt.title('LDA_accuracy_score_'+dataset)
    # plt.xlabel('K')
    # plt.ylabel('Mean Accuracy')
    # plt.grid()
    
    # plt.plot(score_df.reset_index()['index'],score_df['test_score'],label='test_score',color='red')
    # plt.plot(score_df.reset_index()['index'],score_df['train_score'],label='train_score',color='green')
    # # plt.bar(range(2,result_df.shape[0]+2),result_df['mean_kurtosis'],align='center',label='mean kurtosis')
    
    # # LOGGER.info('ica max kurtosis on {}: k={}'.format(dataset,result_df.idxmax(axis=0)['mean_kurtosis']))
    
    # plt.savefig('plots/'+'lda_score_'+dataset+'.png')


def plot_learning_curve(train_scores, test_scores, train_sizes,name_str, filename='l_curve.png'):
    #source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.clf()
    plt.figure()
    title_str = name_str
    plt.title('Learning Curve ({})'.format(title_str))
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig('plots/' + '_'.join([name_str,filename]))

def plot_validation_curve(train_scores, test_scores, param, param_range, name_str,filename='v_curve.png'):
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.clf()
    plt.title("Validation {}".format(name_str))
    plt.xlabel(param)
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    param_range = [str(x) for i,x in enumerate(param_range)]
    plt.plot(param_range, train_scores_mean, label="Training score",
                color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.2,
                    color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.2,
                    color="navy", lw=lw)
    plt.legend(loc="best")
    plt.savefig('plots/' + '_'.join([name_str,filename]))

def run_NN(X,y,model,name):
    
    CV = 5
    N_JOBS = 10
    
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, scoring='accuracy', cv=CV, n_jobs=N_JOBS, shuffle=True)
    plot_learning_curve(train_scores, test_scores, train_sizes,name)
    
    
def run_kmeans_2(trainX):
     #find k
    cluster_counts = {
        'wine':3,
        'wage':2,
    }
    model = KMeans()
    visualizer = KElbowVisualizer(model,k=(2,100),metric='calinski_harabasz',timings=True)
    visualizer.fit(X_train)
    visualizer.show()
    plt.tight_layout()
    plt.savefig('plots/km_sl_'+dataset+'.png')
    
    #validation
    model.set_params(n_clusters=cluster_counts[dataset])
    model.fit(X_train)
    score_fns = [
            v_measure_score,
            homogeneity_score,
            completeness_score,
        ]
    cluster_validation_df = pd.DataFrame()
    for score in score_fns:
        cluster_validation_df.loc[score.__name__,'score'] = score(y_test[y_test.columns[0]],model.predict(X_test))
    print(cluster_validation_df)



def run_gmm_2(trainX,dataset):
    model = GaussianMixture(random_state=0)
    # visualizer = KElbowVisualizer(model,k=(2,100),metric='calinski_harabasz',timings=True)
    # visualizer.fit(X_train)
    # visualizer.show()
    # plt.tight_layout()
    # plt.savefig('plots/gm_ch_'+dataset+'.png')
    
    
    score_df = pd.DataFrame()
    
    k_max = 100
    for k in range(2,k_max):
        model.set_params(n_components=k)
        predY = model.fit_predict(trainX)
        score_df.loc[k,'score'] = calinski_harabasz_score(trainX,predY)
        
    plt.clf()
    plt.title("calinski_harabasz_Expectation_Maximization")
    plt.xlabel('k')
    plt.ylabel('score')
    plt.plot(score_df.reset_index()['index'],score_df['score'],label='calinski_harabasz_score')
    plt.legend(loc="best")
    plt.savefig('plots/' + '_'.join(['gm','ch',dataset,'.png']))
    
    
    
def run_16(X_train, X_test, y_train, y_test,dataset):
    LOGGER.info('running 16...')
    
    settings = {
        'wage':{
            'pca':65,
            'ica':92,
            'rp':105,
            'lda':1,
            'kmeans':2,
            'gmm':2,
            'kmeans_ica':83,
            'kmeans_lda':99,
            'gmm_lda':99,
            'gmm_ica':83,
        },
        'wine':{
            'pca':12,
            'ica':12,
            'rp':13,
            'lda':2,
            'kmeans':3,
            'gmm':3,
            'kmeans_lda':99,
            'gmm_lda':99,
        },
    }
    
    score_fns = [
            v_measure_score,
            homogeneity_score,
            completeness_score,
        ]
    
    
    
    pca = PCA(n_components=settings[dataset]['pca'])
    pca.fit(X_train)
    ica = FastICA(n_components=settings[dataset]['ica'])
    ica.fit(X_train)
    rp = SparseRandomProjection(n_components=settings[dataset]['rp'])
    rp.fit(X_train)
    lda = LinearDiscriminantAnalysis(n_components=settings[dataset]['lda'])
    lda.fit(X_train,y_train)
        
    
    plt.clf()
    visualizer = KElbowVisualizer(KMeans(),k=(2,100),metric='calinski_harabasz',timings=True)
    visualizer.fit(pca.transform(X_train))
    # visualizer.show()
    plt.tight_layout()
    plt.savefig('plots/p16/km_pca_'+dataset+'.png')
    # visualizer.poof()
    kmeans = KMeans(n_clusters=settings[dataset]['kmeans'],random_state=0)
    kmeans.fit(pca.transform(X_train))
    cluster_validation_df = pd.DataFrame()
    for score in score_fns:
        cluster_validation_df.loc[score.__name__,'score'] = score(y_test[y_test.columns[0]],kmeans.predict(pca.transform(X_test)))
    # print(cluster_validation_df)
    LOGGER.info('KMeans PCA {}: \n{}'.format(dataset,cluster_validation_df))
             
    plt.clf() 
    visualizer = KElbowVisualizer(KMeans(),k=(2,100),metric='calinski_harabasz',timings=True)
    visualizer.fit(ica.transform(X_train))
    # visualizer.show()
    plt.tight_layout()
    plt.savefig('plots/p16/km_ica_'+dataset+'.png')
    # visualizer.poof()
    kmeans = KMeans(n_clusters=settings[dataset]['kmeans'],random_state=0)
    kmeans.fit(ica.transform(X_train))
    cluster_validation_df = pd.DataFrame()
    for score in score_fns:
        cluster_validation_df.loc[score.__name__,'score'] = score(y_test[y_test.columns[0]],kmeans.predict(ica.transform(X_test)))
    # print(cluster_validation_df)
    LOGGER.info('KMeans ICA {}: \n{}'.format(dataset,cluster_validation_df))
                
    plt.clf()     
    visualizer = KElbowVisualizer(KMeans(),k=(2,100),metric='calinski_harabasz',timings=True)
    visualizer.fit(rp.transform(X_train))
    # visualizer.show()
    plt.tight_layout()
    plt.savefig('plots/p16/km_rp_'+dataset+'.png')
    # visualizer.poof()
    kmeans = KMeans(n_clusters=settings[dataset]['kmeans'],random_state=0)
    kmeans.fit(rp.transform(X_train))
    cluster_validation_df = pd.DataFrame()
    for score in score_fns:
        cluster_validation_df.loc[score.__name__,'score'] = score(y_test[y_test.columns[0]],kmeans.predict(rp.transform(X_test)))
    # print(cluster_validation_df)
    LOGGER.info('KMeans RP {}: \n{}'.format(dataset,cluster_validation_df))
                
    plt.clf()       
    visualizer = KElbowVisualizer(KMeans(),k=(2,100),metric='calinski_harabasz',timings=True)
    visualizer.fit(lda.transform(X_train))
    # visualizer.show()
    plt.tight_layout()
    plt.savefig('plots/p16/km_lda_'+dataset+'.png')
    # visualizer.poof()
    kmeans = KMeans(n_clusters=settings[dataset]['kmeans'],random_state=0)
    kmeans.fit(lda.transform(X_train))
    cluster_validation_df = pd.DataFrame()
    for score in score_fns:
        cluster_validation_df.loc[score.__name__,'score'] = score(y_test[y_test.columns[0]],kmeans.predict(lda.transform(X_test)))
    # print(cluster_validation_df)
    LOGGER.info('KMeans LDA {}: \n{}'.format(dataset,cluster_validation_df))
    
    
    
    
    
    
    
    
    
    
    gmm = GaussianMixture(random_state=0)
    score_df = pd.DataFrame()
    k_max = 100
    for k in range(2,k_max):
        gmm.set_params(n_components=k)
        predY = gmm.fit_predict(pca.transform(X_train))
        score_df.loc[k,'score'] = calinski_harabasz_score(pca.transform(X_train),predY)   
    LOGGER.info('gmm pca max score on {}: k={}'.format(dataset,score_df.idxmax(axis=0)['score']))
    plt.clf()
    plt.title("calinski_harabasz_Expectation_Maximization")
    plt.xlabel('k')
    plt.ylabel('score')
    plt.plot(score_df.reset_index()['index'],score_df['score'],label='calinski_harabasz_score')
    plt.legend(loc="best")
    plt.savefig('plots/p16/' + '_'.join(['gm','pca',dataset,'.png']))
    gmm = GaussianMixture(n_components=settings[dataset]['gmm'],random_state=0)
    gmm.fit(pca.transform(X_train))
    cluster_validation_df = pd.DataFrame()
    for score in score_fns:
        cluster_validation_df.loc[score.__name__,'score'] = score(y_test[y_test.columns[0]],gmm.predict(pca.transform(X_test)))
    LOGGER.info('GMM PCA {}: \n{}'.format(dataset,cluster_validation_df)) 
    
    gmm = GaussianMixture(random_state=0)
    score_df = pd.DataFrame()
    k_max = 100
    for k in range(2,k_max):
        gmm.set_params(n_components=k)
        predY = gmm.fit_predict(ica.transform(X_train))
        score_df.loc[k,'score'] = calinski_harabasz_score(ica.transform(X_train),predY)   
    LOGGER.info('gmm ica max score on {}: k={}'.format(dataset,score_df.idxmax(axis=0)['score']))
    plt.clf()
    plt.title("calinski_harabasz_Expectation_Maximization")
    plt.xlabel('k')
    plt.ylabel('score')
    plt.plot(score_df.reset_index()['index'],score_df['score'],label='calinski_harabasz_score')
    plt.legend(loc="best")
    plt.savefig('plots/p16/' + '_'.join(['gm','ica',dataset,'.png']))
    gmm = GaussianMixture(n_components=settings[dataset]['gmm'],random_state=0)
    gmm.fit(ica.transform(X_train))
    cluster_validation_df = pd.DataFrame()
    for score in score_fns:
        cluster_validation_df.loc[score.__name__,'score'] = score(y_test[y_test.columns[0]],gmm.predict(ica.transform(X_test)))
    LOGGER.info('GMM ICA {}: \n{}'.format(dataset,cluster_validation_df))
    
    gmm = GaussianMixture(random_state=0)
    score_df = pd.DataFrame()
    k_max = 100
    for k in range(2,k_max):
        gmm.set_params(n_components=k)
        predY = gmm.fit_predict(rp.transform(X_train))
        score_df.loc[k,'score'] = calinski_harabasz_score(rp.transform(X_train),predY)   
    LOGGER.info('gmm rp max score on {}: k={}'.format(dataset,score_df.idxmax(axis=0)['score']))
    plt.clf()
    plt.title("calinski_harabasz_Expectation_Maximization")
    plt.xlabel('k')
    plt.ylabel('score')
    plt.plot(score_df.reset_index()['index'],score_df['score'],label='calinski_harabasz_score')
    plt.legend(loc="best")
    plt.savefig('plots/p16/' + '_'.join(['gm','rp',dataset,'.png']))
    gmm = GaussianMixture(n_components=settings[dataset]['gmm'],random_state=0)
    gmm.fit(rp.transform(X_train))
    cluster_validation_df = pd.DataFrame()
    for score in score_fns:
        cluster_validation_df.loc[score.__name__,'score'] = score(y_test[y_test.columns[0]],gmm.predict(rp.transform(X_test)))
    LOGGER.info('GMM RP {}: \n{}'.format(dataset,cluster_validation_df))
    
    gmm = GaussianMixture(random_state=0)
    score_df = pd.DataFrame()
    k_max = 100
    for k in range(2,k_max):
        gmm.set_params(n_components=k)
        predY = gmm.fit_predict(lda.transform(X_train))
        score_df.loc[k,'score'] = calinski_harabasz_score(lda.transform(X_train),predY)   
    LOGGER.info('gmm lda max score on {}: k={}'.format(dataset,score_df.idxmax(axis=0)['score']))
    plt.clf()
    plt.title("calinski_harabasz_Expectation_Maximization")
    plt.xlabel('k')
    plt.ylabel('score')
    plt.plot(score_df.reset_index()['index'],score_df['score'],label='calinski_harabasz_score')
    plt.legend(loc="best")
    plt.savefig('plots/p16/' + '_'.join(['gm','lda',dataset,'.png']))
    gmm = GaussianMixture(n_components=settings[dataset]['gmm'],random_state=0)
    gmm.fit(lda.transform(X_train))
    cluster_validation_df = pd.DataFrame()
    for score in score_fns:
        cluster_validation_df.loc[score.__name__,'score'] = score(y_test[y_test.columns[0]],gmm.predict(lda.transform(X_test)))
    LOGGER.info('GMM LDA {}: \n{}'.format(dataset,cluster_validation_df))
                
                
                
    ## ICA TEST DO NOT RUN     
    # plt.clf() 
    # visualizer = KElbowVisualizer(KMeans(),k=(2,100),metric='calinski_harabasz',timings=True)
    # visualizer.fit(ica.transform(X_train))
    # # visualizer.show()
    # plt.tight_layout()
    # plt.savefig('plots/p16/km_ica_'+dataset+'.png')
    # visualizer.poof()
    # kmeans = KMeans(n_clusters=settings[dataset]['kmeans_ica'],random_state=0)
    # kmeans.fit(ica.transform(X_train))
    # cluster_validation_df = pd.DataFrame()
    # for score in score_fns:
    #     cluster_validation_df.loc[score.__name__,'score'] = score(y_test[y_test.columns[0]],kmeans.predict(ica.transform(X_test)))
    # # print(cluster_validation_df)
    # LOGGER.info('KMeans ICA {} with clusters {}: \n{}'.format(dataset,settings[dataset]['kmeans_ica'],cluster_validation_df))
    
    # gmm = GaussianMixture(n_components=settings[dataset]['gmm_ica'],random_state=0)
    # gmm.fit(ica.transform(X_train))
    # cluster_validation_df = pd.DataFrame()
    # for score in score_fns:
    #     cluster_validation_df.loc[score.__name__,'score'] = score(y_test[y_test.columns[0]],gmm.predict(ica.transform(X_test)))
    # LOGGER.info('GMM ICA {} with clusters {}: \n{}'.format(dataset,settings[dataset]['gmm_ica'],cluster_validation_df))
                
    
    

    
def run_nn_2(X_train, X_test, y_train, y_test,dataset):
    LOGGER.info('running NN...')
    
    settings = {
        'wage':{
            'pca':65,
            'ica':92,
            'rp':105,
            'lda':1,
            'kmeans':2,
            'gmm':2,
            'kmeans_ica':83,
            'kmeans_lda':99,
            'gmm_lda':99,
            'gmm_ica':83,
            'nn':{
                'iter':200,
                'hls':1000,
                'alpha':.0001,
            },
        },
        'wine':{
            'pca':12,
            'ica':12,
            'rp':13,
            'lda':2,
            'kmeans':3,
            'gmm':3,
            'kmeans_lda':99,
            'gmm_lda':99,
            'nn':{
                'iter':200,
                'hls':800,
                'alpha':.1,
            },
        },
    }
    
    LOGGER.info('NN OG...')
    nn = MLPClassifier(max_iter=settings[dataset]['nn']['iter'],hidden_layer_sizes=settings[dataset]['nn']['hls'],alpha=settings[dataset]['nn']['alpha'])
    nn_check(X_train,X_test,y_train,y_test,nn,'OG')
    nn_epochs(X_train.to_numpy(),X_test.to_numpy(),y_train,y_test,nn,'OG')

    LOGGER.info('NN PCA...')
    pca = PCA(n_components=settings[dataset]['pca'],random_state=0)
    X_train_transformed = pca.fit_transform(X_train)
    X_test_transformed = pca.transform(X_test)
    nn = MLPClassifier(max_iter=settings[dataset]['nn']['iter'],hidden_layer_sizes=settings[dataset]['nn']['hls'],alpha=settings[dataset]['nn']['alpha'])
    nn_check(X_train_transformed,X_test_transformed,y_train,y_test,nn,'PCA')
    nn_epochs(X_train_transformed,X_test_transformed,y_train,y_test,nn,'PCA')
    
    
    LOGGER.info('NN ICA...')
    ica = FastICA(n_components=settings[dataset]['ica'],random_state=0)
    X_train_transformed = ica.fit_transform(X_train)
    X_test_transformed = ica.transform(X_test)
    nn = MLPClassifier(max_iter=settings[dataset]['nn']['iter'],hidden_layer_sizes=settings[dataset]['nn']['hls'],alpha=settings[dataset]['nn']['alpha'])
    nn_check(X_train_transformed,X_test_transformed,y_train,y_test,nn,'ICA')
    nn_epochs(X_train_transformed,X_test_transformed,y_train,y_test,nn,'ICA')
    
    LOGGER.info('NN RP...')
    rp = SparseRandomProjection(n_components=settings[dataset]['rp'],random_state=0)
    X_train_transformed = rp.fit_transform(X_train)
    X_test_transformed = rp.transform(X_test)
    nn = MLPClassifier(max_iter=settings[dataset]['nn']['iter'],hidden_layer_sizes=settings[dataset]['nn']['hls'],alpha=settings[dataset]['nn']['alpha'])
    nn_check(X_train_transformed,X_test_transformed,y_train,y_test,nn,'RP')
    nn_epochs(X_train_transformed,X_test_transformed,y_train,y_test,nn,'RP')
    
    LOGGER.info('NN LDA...')
    lda = LinearDiscriminantAnalysis(n_components=settings[dataset]['lda'])
    X_train_transformed = lda.fit_transform(X_train,y_train)
    X_test_transformed = lda.transform(X_test)
    nn = MLPClassifier(max_iter=settings[dataset]['nn']['iter'],hidden_layer_sizes=settings[dataset]['nn']['hls'],alpha=settings[dataset]['nn']['alpha'])
    nn_check(X_train_transformed,X_test_transformed,y_train,y_test,nn,'LDA')
    nn_epochs(X_train_transformed,X_test_transformed,y_train,y_test,nn,'LDA')
        
    kmeans = KMeans(n_clusters=settings[dataset]['kmeans'],random_state=0)
    X_train_transformed = kmeans.fit_transform(X_train)
    X_test_transformed = kmeans.transform(X_test)
    nn = MLPClassifier(max_iter=settings[dataset]['nn']['iter'],hidden_layer_sizes=settings[dataset]['nn']['hls'],alpha=settings[dataset]['nn']['alpha'])
    nn_check(X_train_transformed,X_test_transformed,y_train,y_test,nn,'KMEANS')
    nn_epochs(X_train_transformed,X_test_transformed,y_train,y_test,nn,'KMEANS')
    
    gmm = GaussianMixture(n_components=settings[dataset]['gmm'],random_state=0)
    gmm.fit(X_train)
    X_train_transformed = gmm.predict_proba(X_train)
    X_test_transformed = gmm.predict_proba(X_test)
    # X_train_transformed = gmm.predict(X_train)
    # X_test_transformed = gmm.predict(X_test)
    # print(X_train_transformed)
    # print(X_test_transformed)
    nn = MLPClassifier(max_iter=settings[dataset]['nn']['iter'],hidden_layer_sizes=settings[dataset]['nn']['hls'],alpha=settings[dataset]['nn']['alpha'])
    nn_check(X_train_transformed,X_test_transformed,y_train,y_test,nn,'GMM')
    nn_epochs(X_train_transformed,X_test_transformed,y_train,y_test,nn,'GMM')
    

def nn_check(X_train, X_test, y_train, y_test,model,name):
    st = time.time()
    model.fit(X_train,y_train)
    fit_time = time.time() - st
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_acc = accuracy_score(y_train,y_pred_train)
    test_acc = accuracy_score(y_test,y_pred_test)
    
    LOGGER.info('NN {} | time: {} | acc test: {} , train: {}'.format(name,fit_time,test_acc,train_acc))
    

def nn_epochs(X_train, X_test, y_train, y_test,model,name):
    #source: https://stackoverflow.com/questions/46912557/is-it-possible-to-get-test-scores-for-each-iteration-of-mlpclassifier
    
    
    classes = np.unique(y_train)
    batch_count = 10
    batch_size = int(X_train.shape[0]/batch_count)
    epochs = 1000
    scores_train = []
    scores_test = []
    for epoch in range(epochs):
        print('epoch:{}'.format(epoch))
        model.partial_fit(X_train,y_train,classes=classes)
        data_perm = np.random.permutation(X_train.shape[0])
        index = 0
        count = 0
        while index < X_train.shape[0]:
            print('batch:{}'.format(count))
            indicies = data_perm[index:index+batch_size]
            model.partial_fit(X_train[indicies],y_train.iloc[indicies],classes=classes)
            index += batch_size
            count += 1
            if index > X_train.shape[0]: break
        
        scores_train.append(model.score(X_train,y_train))
        scores_test.append(model.score(X_test,y_test))
        
    plt.clf()
    plt.plot(scores_train, color='orange', alpha=1.0, label='Train')
    plt.plot(scores_test, color='green', alpha=1.0, label='Test')
    plt.title('Accuracy vs. epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy Score')
    plt.legend(loc='upper left')
    plt.savefig('plots/nn/mlp_{}.png'.format(name))   
    
    

def run_dataset(dataset,nn):
    LOGGER.info('dataset: {}'.format(dataset))
    df_X,df_y = get_data_df(dataset)
        
    # principal_component_count = {
    #     'wine':12,
    #     'wage':65,
    # }
    # n_components = principal_component_count[dataset]
    
    # pca_df,km_df,gm_df,pca,pca_km,pca_gm,proj_loss = run_pca(df_X,df_y,n_components=n_components)
    # plot_eigen(pca_df,'pca_eigenvalues_'+dataset,'components','value')
    # plot_scores(km_df,'pca_kmeans_'+str(n_components)+'_'+dataset,'k','score')
    # plot_scores(gm_df,'pca_gm_'+str(n_components)+'_'+dataset,'k','score')
    # plot_projection_loss(proj_loss,dataset+'_Reconstructed_Projection_Loss','component','mean squared error')
    
    # independent_component_count = {
    #     'wine':7,
    #     'wage':19,
    # }
    # n_components = independent_component_count[dataset]
    
    # ica_df,km_df,gm_df,ica,ica_km,ica_gm = run_ica(df_X,df_y,n_components=n_components)
    # plot_kurt(ica_df,'ica_kurt_'+dataset,'components','value')
    # plot_scores(km_df,'ica_kmeans_'+str(n_components)+'_'+dataset,'k','score')
    # plot_scores(gm_df,'ica_gm_'+str(n_components)+'_'+dataset,'k','score')
    
    
    # independent_component_count = {
    #     'wine':7,
    #     'wage':19,
    # }
    # n_components = independent_component_count[dataset]
    # km_df,gm_df,rp,rp_km,rp_gm = run_rp(df_X,df_y,n_components=n_components)
    # plot_scores(km_df,'rp_kmeans_'+str(n_components)+'_'+dataset,'k','score')
    # plot_scores(gm_df,'rp_gm_'+str(n_components)+'_'+dataset,'k','score')
    
    
    
    # independent_component_count = {
    #     'wine':7,
    #     'wage':19,
    # }
    # n_components = independent_component_count[dataset]
    # km_df,gm_df,lda,lda_km,lda_gm = run_lda(df_X,df_y,n_components=n_components,name='lda_lc_'+dataset)
    # plot_scores(km_df,'lda_kmeans_'+str(n_components)+'_'+dataset,'k','score')
    # plot_scores(gm_df,'lda_gm_'+str(n_components)+'_'+dataset,'k','score')
    
    
    
    split_ratio = 0.33
    X_train, X_test, y_train, y_test = train_test_split(df_X,df_y,test_size=split_ratio,random_state=0)
    LOGGER.debug('train test split: {}'.format(split_ratio))
    
    # kmeans_df,choose_df,km_model = run_kmeans(X_train, X_test, y_train, y_test)
    # plot_scores(kmeans_df,'kmeans_validate_'+dataset,'k','score')
    # plot_scores(choose_df,'kmeans_choose_'+dataset,'k','score')
    
    #find k
    cluster_counts = {
        'wine':3,
        'wage':2,
    }
    model = KMeans()
    visualizer = KElbowVisualizer(model,k=(2,100),metric='calinski_harabasz',timings=True)
    visualizer.fit(X_train)
    visualizer.show()
    plt.tight_layout()
    plt.savefig('plots/km_sl_'+dataset+'.png')
    
    #validation
    model.set_params(n_clusters=cluster_counts[dataset])
    model.fit(X_train)
    score_fns = [
            v_measure_score,
            homogeneity_score,
            completeness_score,
        ]
    cluster_validation_df = pd.DataFrame()
    for score in score_fns:
        cluster_validation_df.loc[score.__name__,'score'] = score(y_test[y_test.columns[0]],model.predict(X_test))
    print(cluster_validation_df)
    
    
    # run_gmm_2(X_train,dataset)
    cluster_counts = {
        'wine':3,
        'wage':2,
    }
    #validation
    model = GaussianMixture(random_state=0)
    model.set_params(n_components=cluster_counts[dataset])
    model.fit(X_train)
    score_fns = [
            v_measure_score,
            homogeneity_score,
            completeness_score,
        ]
    cluster_validation_df = pd.DataFrame()
    for score in score_fns:
        cluster_validation_df.loc[score.__name__,'score'] = score(y_test[y_test.columns[0]],model.predict(X_test))
    print(cluster_validation_df)
    
    
    run_ica_2(X_train,dataset)
    
    run_rp_2(X_train,dataset)
    
    run_lda_2(X_train, X_test, y_train, y_test,dataset)
    
    run_16(X_train, X_test, y_train, y_test,dataset)
    
    run_nn_2(X_train, X_test, y_train, y_test,dataset)
    
    # gm_df,gm_model = run_gm(X_train, X_test, y_train, y_test)
    # plot_scores(gm_df,'gm_'+dataset,'k','score')
    
    
    # run_NN(df_X,df_y,nn,'nn_og')
    
    # pca_x = pca.transform(df_X)
    # run_NN(pca_x,df_y,nn,'nn_pca')
    
    # ica_x = ica.transform(df_X)
    # run_NN(ica_x,df_y,nn,'nn_ica_'+dataset)
    
    # rp_x = rp.transform(df_X)
    # run_NN(rp_x,df_y,nn,'nn_rp_'+dataset)
    
    # lda_x = lda.transform(df_X)
    # run_NN(lda_x,df_y,nn,'nn_lda_'+dataset)
    
    # km_x = km_model.transform(df_X)
    # run_NN(lda_x,df_y,nn,'nn_km_'+dataset)
    
    # gm_x = gm_model.predict(df_X)
    # run_NN(lda_x,df_y,nn,'nn_gm_'+dataset)
    
    
    
    # -----------------------------------------------------------------
    # pca_km_x = pca_km.transform(pca_x)
    # run_NN(pca_x,df_y,nn,'nn_pca_km')
    # pca_gm_x = pca_gm.transform(pca_x)
    # run_NN(pca_x,df_y,nn,'nn_pca_gm')
    
    
    

def run():
    LOGGER.info('running...')
    LOGGER.info('NOTE: this will take a long time, there is a lot to run')
    
    
    nn = MLPClassifier(max_iter=200,hidden_layer_sizes=800,alpha=.1)
    run_dataset('wine',nn)
    
    nn = MLPClassifier(max_iter=200,hidden_layer_sizes=1000,alpha=.0001)
    run_dataset('wage',nn)

if __name__ == '__main__':
    LOGGER = create_logger()
    run()