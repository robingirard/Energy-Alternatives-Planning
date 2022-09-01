df.loc[cat_tuple(tuple(len(Index_set.columns) * [slice(None)]), new_x_values[-1]), col] = df.loc[
    cat_tuple(tuple(len(Index_set.columns) * [slice(None)]), x_columns_for_interpolation_input_values), col]
