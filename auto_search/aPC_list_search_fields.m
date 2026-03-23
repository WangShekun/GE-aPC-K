function fields = aPC_list_search_fields()
% aPC_list_search_fields
% -------------------------------------------------------
% List of fields that support scalar-or-vector auto-search.
% Add new fields here when new hyperparameters are introduced.
% -------------------------------------------------------

fields = {
    'Order'          % polynomial chaos order
    'qnorm'          % q-norm truncation
    'red_dim'        % dimensionality reduction target size

    % ===== X-KERNEL (for KPCA / SKPCA input kernel) =====
    'kernel_type'
    'kernel_gamma'
    'kernel_degree'
    'kernel_offset'

    % ===== Y-KERNEL (for supervised KPCA / HSIC) =====
    'label_kernel_type'
    'label_kernel_gamma'
    'label_kernel_degree'
    'label_kernel_offset'

    % Future extension:
    % 'ridge_lambda'
    % 'lasso_alpha'
    % 'poly_trunc_level'
};

end
