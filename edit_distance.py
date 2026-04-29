import numpy as np
import config

# @profile
def custom_word_edit_distance(str1, str2, sub_matrix, del_cost=1.0, ins_cost=1.0,**kwargs):
    if not hasattr(custom_word_edit_distance, "_sub_cache"):
        custom_word_edit_distance._sub_cache = {}
    sub_cache = custom_word_edit_distance._sub_cache

    words1, words2 = str1.split(),str2.split()
    n,m = len(words1), len(words2)
    dp = np.zeros((n + 1, m + 1), dtype=np.float32)
    ops = [[None for j in range(m + 1)] for i in range(n + 1)]  # Matrix to store operations
    if not kwargs:
        if type(next(iter(sub_matrix.keys())))==tuple:
            for i in range(n + 1):
                for j in range(m + 1):
                    if i == 0:
                        dp[i][j] = j * ins_cost
                    elif j == 0:
                        dp[i][j] = i * del_cost
                    elif words1[i - 1] == words2[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1]
                    else:
                        if (words1[i - 1], words2[j - 1]) in sub_matrix:
                            sub_cost = sub_matrix.get((words1[i - 1], words2[j - 1]), 1)
                        elif (words2[j - 1], words1[i - 1]) in sub_matrix:
                            sub_cost = sub_matrix.get((words2[j - 1], words1[i - 1]), 1)
                        else:
                            print(f'no substitution entry for words {words1[i-1]} and {words2[j-1]}')
                        dp[i][j] = min(dp[i - 1][j] + del_cost,
                                       dp[i][j - 1] + ins_cost,
                                       dp[i - 1][j - 1] + sub_cost)
        elif type(next(iter(sub_matrix.keys())))==str:
            for i in range(n + 1):
                for j in range(m + 1):
                    if i == 0:
                        dp[i][j] = j * ins_cost
                    elif j == 0:
                        dp[i][j] = i * del_cost
                    elif words1[i - 1] == words2[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1]
                    else:
                        try:
                            sub_cost = sub_matrix[words1[i - 1]][words2[j - 1]]
                        except:
                            print(f'no substitution entry for words {words1[i-1]} and {words2[j-1]}')
                        dp[i][j] = min(dp[i - 1][j] + del_cost,
                                       dp[i][j - 1] + ins_cost,
                                       dp[i - 1][j - 1] + sub_cost)
    elif 'unique_words_dict' in kwargs:
        unique_words_dict = kwargs['unique_words_dict']

        for i in range(1, n + 1):
            dp[i][0] = i * del_cost
        for j in range(1, m + 1):
            dp[0][j] = j * ins_cost

        ids1 = [unique_words_dict[w] for w in words1]
        ids2 = [unique_words_dict[w] for w in words2]

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if words1[i - 1] == words2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    sub_cost = sub_matrix[ids1[i - 1]][ids2[j - 1]]
                    dp[i][j] = min(
                        dp[i - 1][j] + del_cost,
                        dp[i][j - 1] + ins_cost,
                        dp[i - 1][j - 1] + sub_cost
                    )
    return dp[n][m]

