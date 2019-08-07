def feature_peek(df, column_list):
    for c in column_list:
        cn = df[c][~df[c].isna()]
        print(c)
        print(str(df[c].dtype) + ' | ' +
              str(sum(df[c].isna())) + ' NaNs | ' +
              str(len(cn.unique())) + ' unique')
        if len(cn.unique()) < 10:
            print('Values: ' + str(list(cn.unique())))
        elif df[c].dtype != 'object':
            print('min ' + str(df[c].min()) +
                  ' | med ' + str(df[c].median()) +
                  ' | max ' + str(df[c].max()))
        else:
            print('Sample: ' + str(list(cn.sample(5))))
        if c != column_list[-1]:
            print()
