import random
import pandas as pd
from enum import Enum
import math 

class Behavior(Enum):
    HONEST = 1
    DISHONEST = 2
    UNPREDICTABLE = 3
    
class AccountType(Enum):
    ANONYMOUS = 1
    TEMP_ANONYMOUS = 2
    ALTERNATE = 3
    PUBLIC = 4

class Feature(Enum):
    ZKP = 1
    VERIFIABLE_CREDENTIAL = 2
    GOV_REAL_ID = 3
    CONCURRENT_TRANSACTIONS = 4
    HISTORY_ORIENTED = 5
    RELIABLE = 6
    LINKABILITY = 7
    QUESTIONABLE = 8
    PSUDO_ANONYMOUS = 9
    MULTI_FACTOR_AUTH = 10
    
    
# Define mandatory and optional features for each account type
ACCOUNT_TYPE_FEATURES = {
    AccountType.ANONYMOUS: {
        'mandatory': [Feature.ZKP],
        'optional': [Feature.PSUDO_ANONYMOUS]
    },
    AccountType.TEMP_ANONYMOUS: {
        'mandatory': [Feature.PSUDO_ANONYMOUS],
        'optional': [Feature.HISTORY_ORIENTED, Feature.QUESTIONABLE]
    },
    AccountType.ALTERNATE: {
        'mandatory': [Feature.VERIFIABLE_CREDENTIAL, Feature.RELIABLE],
        'optional': [Feature.ZKP, Feature.LINKABILITY, Feature.MULTI_FACTOR_AUTH]
    },
    
    AccountType.PUBLIC: {
        'mandatory': [Feature.GOV_REAL_ID, Feature.RELIABLE],
        'optional': [Feature.VERIFIABLE_CREDENTIAL, Feature.CONCURRENT_TRANSACTIONS]
    },
}

class User:
    def __init__(self, id):
        self.id = id
        self.trust_score = 0
        self.behavior = self.assign_behavior()
        self.account_type = self.assign_account_type()
        self.features = self.initialize_features()

    def assign_behavior(self):
        return random.choice(list(Behavior))
        
    def assign_account_type(self):
        return random.choice(list(AccountType))

    def initialize_features(self):
        features = ACCOUNT_TYPE_FEATURES[self.account_type]['mandatory'][:]  # Mandatory features
        optional_features = ACCOUNT_TYPE_FEATURES[self.account_type]['optional'][:]
        
        # Each user has a 50% chance of having each optional feature
        for feature in optional_features:
            if random.random() > 0.5:
                features.append(feature)
        return features

    def calculate_feature_score(self):
        if self.account_type == AccountType.PUBLIC:
            return len(self.features) * 40
        elif self.account_type == AccountType.ALTERNATE:
            return len(self.features) * 30
        elif self.account_type == AccountType.TEMP_ANONYMOUS:
            return len(self.features) * 20
        else:
            return len(self.features) * 10

    def update_features(self):
        # Each user has a 20% chance to remove a feature and 30% to add a feature
        # (only optional features can be removed or added)
        optional_features = ACCOUNT_TYPE_FEATURES[self.account_type]['optional']
        for feature in optional_features:
            if feature in self.features and random.random() < 0.2:  # 20% chance to remove a feature
                self.features.remove(feature)
            elif feature not in self.features and random.random() < 0.3:
                self.features.append(feature)
                
                
        # Each user can 10% of the time drop their mandatory features if they are dishonest
        mandatory_features = ACCOUNT_TYPE_FEATURES[self.account_type]['mandatory']
        for feature in mandatory_features:
            if feature in self.features and random.random() < 0.1 and self.behavior == Behavior.DISHONEST:
                self.features.remove(feature)
                
        #TODO: adjust the 10% dishonest user [30%, 50%] (cause we don't know the exact percentage of dishonest users)
            

    # Returns the number of features required for the user's account type
    def required_features(self):
        return len(ACCOUNT_TYPE_FEATURES[self.account_type]['mandatory'])

    def claim(self):
        return self.features  # Returns the features the user claims to have

    # Verify if the user's claim is still valid to it's original account type
    # If not, the user is dishonest according the PIBRP protocol
    def verify_claim(self):
        return all(feature in self.features for feature in ACCOUNT_TYPE_FEATURES[self.account_type]['mandatory'])

    # This is the function that calculates the endorsement score for the user by other users
    def get_endorsement(self, other):
        # If the claim is verified, calculate the endorsement score based on other user's feature score
        if self.verify_claim():
            if self.behavior == Behavior.HONEST:
                endorsement_score = math.log(other.calculate_feature_score() + 1)  # Honest users provide a log endorsement score
            elif self.behavior == Behavior.DISHONEST:
                endorsement_score = -1 * other.calculate_feature_score()  # Dishonest users deduct the feature score
            else:
                endorsement_score = random.choice([1, -1]) * math.log(other.calculate_feature_score() + 1)  # Unpredictable users randomly provide a log endorsement score or deduct the feature score
        else:
            # If the claim is not verified, deduct the user's from the user's trust score
            endorsement_score = -1 * self.calculate_feature_score()  # If the claim is not verified, deduct the feature score

        return endorsement_score

    def update_trust_score(self, endorsement_score):
        self.trust_score += endorsement_score  # The trust score is updated by the endorsement score


class Transaction:
    def __init__(self, user1, user2):
        self.user1 = user1
        self.user2 = user2

class Blockchain:
    def __init__(self, num_users):
        self.users = [User(i) for i in range(num_users)]

    def perform_transactions(self):
        # Randomly select two users for each transaction
        for _ in range(len(self.users) // 2):
            user1, user2 = random.sample(self.users, 2)

            # Calculate the endorsement score for each transaction
            endorsement_score = user1.get_endorsement(user2)
            user1.update_trust_score(endorsement_score) 

            # Update the user's features after every transaction
            user1.update_features()

    def export_results(self):
        data = []
        for user in self.users:
            data.append([user.id, user.behavior.name, user.account_type.name, [f.name for f in user.features], user.trust_score])

        df = pd.DataFrame(data, columns=["UserID", "Behavior", "AccountType", "Features", "TrustScore"])
        df.to_csv("simulation_results_10_perc_feature_drop_by_mal.csv", index=False)

def main():
    blockchain = Blockchain(100000)

    for _ in range(10000):  # Perform 10000 rounds of transactions
        blockchain.perform_transactions()

    blockchain.export_results()

if __name__ == "__main__":
    main()
