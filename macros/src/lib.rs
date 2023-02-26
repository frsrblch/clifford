extern crate proc_macro;

use syn::parse_macro_input;

#[proc_macro]
#[allow(non_snake_case)]
pub fn algebra(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let algebra = parse_macro_input!(input as algebra::Algebra);
    let tokens = algebra.define();
    tokens.into()
}
