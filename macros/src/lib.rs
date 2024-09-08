extern crate proc_macro;

use syn::parse_macro_input;

/// Generates the complete geometric algebra
#[proc_macro]
pub fn algebra(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let algebra = parse_macro_input!(input as algebra::Algebra);
    let tokens = algebra.define();
    tokens.into()
}

/// Generates the a useful fraction of the full algebra to save resources during compilation
#[proc_macro]
#[allow(non_snake_case)]
pub fn lean_algebra(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let algebra = parse_macro_input!(input as algebra::Algebra);
    let tokens = algebra.define_lean();
    tokens.into()
}
