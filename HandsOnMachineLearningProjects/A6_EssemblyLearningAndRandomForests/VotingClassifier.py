from sklearn.datasets import make_moons
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

if __name__ == '__main__':
    x, y = make_moons(n_samples=500, noise=0.30, random_state=42)

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

    voting_clf = VotingClassifier(estimators=[
        ('lr', LogisticRegression()),
        ('rf', RandomForestClassifier(random_state=42)),
        ('svc', SVC(random_state=42))
    ])

    voting_clf.fit(x_train, y_train)

    # now we are going to see the score of the different classifiers

    estimators = voting_clf.named_estimators_
    print(estimators)

    for name, clf in estimators.items():
        print(f'Estimator: {name}, Score: {clf.score(x_test, y_test)}')

    # now we are going to use the predict method
    # if we use the voting classifier predict method we get a 1
    print(f'\nVoting Classifier predict {voting_clf.predict(x_test[:1])}')

    # we get a 1 because 2 of 3 classifier predict a 1, so that's it
    print(
        f'All ClassifierPredictions {[clf.predict(x_test[:1]).tolist() for name, clf in voting_clf.named_estimators_.items()]} \n')

    # and now this is the score using the voting classifier,
    # as we can see his score is better than the classifier scoring alone
    print(f'Voting Classifier Score: {voting_clf.score(x_test, y_test)}')

    # if we are using classifiers that contains predict_proba method
    # is good to use it  because we can get better performance to predict values
    # so in this case we only have to set the probability parameter so SVC
    # and after that our votingClassifier must have the voting type as soft validation to use the probability methods

    voting_clf.voting = 'soft'
    voting_clf.named_estimators['svc'].probability = True

    voting_clf.fit(x_train, y_train)

    print(f'Voting Classifier Score After Used Probabilities: {voting_clf.score(x_test, y_test)}')
