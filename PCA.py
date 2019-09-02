import matplotlib.pyplot as plt
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



# load data
Data = pandas.read_csv("iris_data.csv")



#put feature value in x then Standardizing and put target value in y
features1 = ['sepal length', 'sepal width', 'petal length', 'petal width']
name = ['target']
value_of_feature = Data.loc[:, features1].values
value_of_feature_standard = StandardScaler().fit_transform(value_of_feature)
value_of_target = Data.loc[:,name].values




#find variance_ratio of pca
Pca_create = PCA(n_components=4)
Pca_fit_transform = Pca_create.fit_transform(value_of_feature_standard)
variance_ratio = Pca_create.explained_variance_ratio_


#plot the variance of four feature
plt.bar([1,2,3,4],list(variance_ratio),color='c')
plt.xlabel('Principal Components')
plt.ylabel('Variance Ratio for four feature')
plt.xticks([1,2,3,4],['PC-first','PC-second','PC-third ','PC-fourth'])
plt.show()


#we selecte first and second max variance because this feature is better
Pca_create = PCA(n_components=2)
Pca_fit_transform = Pca_create.fit_transform(value_of_feature_standard)
variance_ratio = Pca_create.explained_variance_ratio_
plt.bar([1,2],list(variance_ratio),color='c')
plt.xlabel('Principal Components')
plt.ylabel('Variance Ratio for two feature')
plt.xticks([1,2],['PC-first','PC-second'])
plt.show()




#find lable for PCA-first and PCA-second and plot them
#I see the feauter is better for three flower because they don't dependent each other
data_for_first_and_second_PCA = pandas.DataFrame(data = Pca_fit_transform , columns = ['PC-first', 'PC-second'])
data_after_pca = pandas.concat([data_for_first_and_second_PCA, Data[['target']]], axis = 1)
print("Iris Data after demation reduction:\n ",data_after_pca)


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('PC-first')
ax.set_ylabel('PC-second')
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['c', 'm', 'b']
for target, color in zip(targets,colors):
    target_index = data_after_pca['target'] == target
    ax.scatter(data_after_pca.loc[target_index, 'PC-first'], data_after_pca.loc[target_index, 'PC-second'], c = color)
ax.legend(targets)
ax.grid()
plt.show()
