from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss

def get_stats(y_test, y_pred, probabilities):
    acc_score = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    lg_loss = log_loss(y_test, probabilities)
    print(f'Log Loss: {lg_loss}, accuracy score: {acc_score}, precision: {precision}, recall: {recall}')
    print('F1 Score: ' + str((2*precision*recall)/(precision+recall)))