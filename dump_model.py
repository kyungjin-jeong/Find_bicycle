"""================================== Dump Model ==================================="""
"""
df_model_score = pd.DataFrame({'model': [upsampled, undersampled, lg_smote,
                                         dt_upsampled, dt_undersampled, dt_smote],
                               'score': [metrics.accuracy_score(Y_test, lg_upsampled_pred),metrics.accuracy_score(Y_test, undersampled_pred), metrics.accuracy_score(Y_test, lg_smote_pred),metrics.accuracy_score(Y_test, upsampled_pred),metrics.accuracy_score(Y_test, undersampled_pred),metrics.accuracy_score(Y_test, smote_pred)],
                               'type': ['lg','lg','lg','dt','dt','dt']
                              })
                  
max(df_model_score['score'])
    
selected_model = ''
selected_type = ''
for i in df_model_score.index:
    if df_model_score['score'][i] == max(df_model_score['score']):
        print(df_model_score['model'][i])
        selected_model = df_model_score['model'][i]
        selected_type = df_model_score['type'][i]
"""

# Dump Decision Tree - Downsampled
import joblib 
joblib.dump(dt_undersampled, '/Users/kyungjin/find_bicycle/final_model.pkl')
print("Model dumped!")

# Logistic regression
if type == 'lg':
    model_columns = cols
# Decision Tree
else:
    model_columns = predictors
    
model_columns

joblib.dump(model_columns, '/Users/kyungjin/find_bicycle/final_model_columns.pkl')
print("Models columns dumped!")
