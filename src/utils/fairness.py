import numpy as np
def calculate_average_odds_difference(TP_male , TN_male, FN_male,FP_male, TP_female , TN_female , FN_female,  FP_female):
    # TPR_male = TP_male/(TP_male+FN_male)
    # TPR_female = TP_female/(TP_female+FN_female)
    # FPR_male = FP_male/(FP_male+TN_male)
    # FPR_female = FP_female/(FP_female+TN_female)
    # average_odds_difference = abs(abs(TPR_male - TPR_female) + abs(FPR_male - FPR_female))/2
    FPR_diff = calculate_FPR_difference(TP_male, TN_male, FN_male,FP_male, TP_female , TN_female , FN_female,  FP_female)
    TPR_diff = calculate_TPR_difference(TP_male , TN_male, FN_male,FP_male, TP_female , TN_female , FN_female,  FP_female)
    average_odds_difference = (FPR_diff + TPR_diff)/2
    #print("average_odds_difference",average_odds_difference)
    return round(average_odds_difference,4)
def calculate_Disparate_Impact(TP_male , TN_male, FN_male,FP_male, TP_female , TN_female , FN_female,  FP_female):
    P_male = (TP_male + FP_male+1)/(TP_male + TN_male + FN_male + FP_male+1)
    P_female = (TP_female + FP_female+1)/(TP_female + TN_female + FN_female +  FP_female+1)
    DI = (P_female/P_male)
    return round((1 - abs(DI)),4)

def calculate_SPD(TP_male , TN_male, FN_male,FP_male, TP_female , TN_female , FN_female,  FP_female):
    P_male = (TP_male + FP_male+1)/(TP_male + TN_male + FN_male + FP_male+1)
    P_female = (TP_female + FP_female+1) /(TP_female + TN_female + FN_female +  FP_female+1)
    SPD = (P_female - P_male)
    return round(abs(SPD),4)


def calculate_equal_opportunity_difference(TP_male , TN_male, FN_male,FP_male, TP_female , TN_female , FN_female,  FP_female):
    # TPR_male = TP_male/(TP_male+FN_male)
    # TPR_female = TP_female/(TP_female+FN_female)
    # equal_opportunity_difference = abs(TPR_male - TPR_female)
    #print("equal_opportunity_difference:",equal_opportunity_difference)
    return calculate_TPR_difference(TP_male , TN_male, FN_male,FP_male, TP_female , TN_female , FN_female,  FP_female)

def calculate_TPR_difference(TP_male , TN_male, FN_male,FP_male, TP_female , TN_female , FN_female,  FP_female):
    TPR_male = (TP_male+1)/(TP_male+FN_male+1)
    TPR_female = (TP_female+1)/(TP_female+FN_female+1)
    # print("TPR_male:",TPR_male,"TPR_female:",TPR_female)
    diff = (TPR_male - TPR_female)
    return round(diff,4)

def calculate_FPR_difference(TP_male , TN_male, FN_male,FP_male, TP_female , TN_female , FN_female,  FP_female):
    FPR_male = (FP_male+1)/(FP_male+TN_male+1)
    FPR_female = (FP_female+1)/(FP_female+TN_female+1)
    # print("FPR_male:",FPR_male,"FPR_female:",FPR_female)
    diff = (FPR_female - FPR_male)
    return round(diff,4)

def _fairness_metrics(TP_male , TN_male, FN_male,FP_male, TP_female , TN_female , FN_female,  FP_female):
    EOD = calculate_TPR_difference(TP_male , TN_male, FN_male,FP_male, TP_female , TN_female , FN_female,  FP_female)
    SPD = calculate_SPD(TP_male , TN_male, FN_male,FP_male, TP_female , TN_female , FN_female,  FP_female)
    DI = calculate_Disparate_Impact(TP_male , TN_male, FN_male,FP_male, TP_female , TN_female , FN_female,  FP_female)
    AOD = calculate_average_odds_difference(TP_male , TN_male, FN_male,FP_male, TP_female , TN_female , FN_female,  FP_female)
    return AOD, EOD, SPD, DI

def fairness_measure_global(TP_a, TN_a, FN_a, FP_a, TP_b, TN_b, FN_b, FP_b, measure='SPD'):
    # (neg_fit + pos_fit, AOD, EOD, SPD, DI)
    f_score = 0
    if measure == 'DI':
        f_score = calculate_Disparate_Impact(TP_a, TN_a, FN_a, FP_a, TP_b, TN_b, FN_b, FP_b)
    elif measure == 'SPD':
        f_score = calculate_SPD(TP_a, TN_a, FN_a, FP_a, TP_b, TN_b, FN_b, FP_b)
    elif measure == 'EOD':
        f_score = calculate_equal_opportunity_difference(TP_a, TN_a, FN_a, FP_a, TP_b, TN_b, FN_b, FP_b)
    elif measure == 'AOD':
        f_score = calculate_average_odds_difference(TP_a, TN_a, FN_a, FP_a, TP_b, TN_b, FN_b, FP_b)
    # else:
    return abs(f_score)

def _fairness_metrics_global(TP_male, TN_male, FN_male, FP_male, TP_female, TN_female, FN_female, FP_female):
    EOD = calculate_equal_opportunity_difference(TP_male, TN_male, FN_male, FP_male, TP_female, TN_female,
                                                 FN_female, FP_female)
    SPD = calculate_SPD(TP_male, TN_male, FN_male, FP_male, TP_female, TN_female, FN_female, FP_female)
    DI = calculate_Disparate_Impact(TP_male, TN_male, FN_male, FP_male, TP_female, TN_female, FN_female, FP_female)
    AOD = calculate_average_odds_difference(TP_male, TN_male, FN_male, FP_male, TP_female, TN_female, FN_female,
                                            FP_female)
    return abs(AOD), abs(EOD), abs(SPD), abs(DI)

def _aggregate_val_global(TP, TN, FN, FP, funct=None):
    if funct is None:
        return np.mean(TP), np.mean(TN), np.mean(FN), np.mean(FP)
    else:
        return funct(TP), funct(TN), funct(FN), funct(FP)

def _fairness_sub_global(test_labels_, prediction_, sensitive_attrib, group_a, group_b, sub_group=None, measure='DI'):
    # print('fitness_dict: ', fitness_dict)

    # group_a, group_b = self.group_a, self.group_b
    test_labels = test_labels_.copy()
    prediction = prediction_.copy()

    if sub_group != None:
        #group_a, group_b = self.group_a, self.group_b  # _get_subgroup_names_as_binary(sensitive_attrib)

        # if prediction.shape[1] > 1:
        #    prediction = np.argmax(prediction,axis=1)
        # if test_labels.shape[1] > 1:
        #    test_labels = np.argmax(test_labels,axis=1)
        TP_a, TN_a, FN_a, FP_a = [], [], [], []
        TP_b, TN_b, FN_b, FP_b = [], [], [], []
        for sub_a in group_a:
            indices = np.where(sensitive_attrib == sub_a)[0]
            pred = prediction.copy()[indices]
            y_true = test_labels.copy()[indices]
            TP_1 = ((y_true == pred) & (y_true == 1)).sum()
            TP_0 = ((y_true == pred) & (y_true == 0)).sum()
            FP_0 = ((y_true != pred) & (pred == 0)).sum()
            FP_1 = ((y_true != pred) & (pred == 1)).sum()

            TP_a.append(TP_0)
            TN_a.append(TP_1)
            FN_a.append(FP_1)
            FP_a.append(FP_0)
        for sub_b in group_b:
            indices = np.where(sensitive_attrib == sub_b)[0]
            pred = prediction.copy()[indices]
            y_true = test_labels.copy()[indices]

            TP_1 = ((y_true == pred) & (y_true == 1)).sum()
            TP_0 = ((y_true == pred) & (y_true == 0)).sum()
            FP_0 = ((y_true != pred) & (pred == 0)).sum()
            FP_1 = ((y_true != pred) & (pred == 1)).sum()

            TP_b.append(TP_1)
            TN_b.append(TP_0)
            FN_b.append(FP_0)
            FP_b.append(FP_1)
        TP_a, TN_a, FN_a, FP_a = _aggregate_val_global(TP_a, TN_a, FN_a, FP_a, funct=np.mean)
        TP_b, TN_b, FN_b, FP_b = _aggregate_val_global(TP_b, TN_b, FN_b, FP_b, funct=np.mean)

        fairness_val = fairness_measure_global(TP_a, TN_a, FN_a, FP_a, TP_b, TN_b, FN_b, FP_b, measure=measure)
    else:
        TP_1 = ((test_labels == prediction) & (test_labels == 1)).sum()
        TP_0 = ((test_labels == prediction) & (test_labels == 0)).sum()

        FP_0 = ((test_labels != prediction) & (prediction == 0)).sum()
        FP_1 = ((test_labels != prediction) & (prediction == 1)).sum()

        TN_0 = TP_1
        TN_1 = TP_0

        FN_0 = FP_1
        FN_1 = FP_0
        # TP_0, TP_1, FP_0, FP_1, FN_0, FN_1

        # fairness_val = _fairness_metrics(TP_0,TN_0,FN_0,FP_0,TP_1,TN_1,FN_1, FP_1)
        fairness_val = fairness_measure_global(TP_0, TN_0, FN_0, FP_0, TP_1, TN_1, FN_1, FP_1, measure=measure)
    return fairness_val