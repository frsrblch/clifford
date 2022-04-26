#![feature(type_alias_impl_trait)]

extern crate proc_macro;

mod code_gen;
mod types;

use syn::parse_macro_input;
use types::Algebra;

#[proc_macro]
pub fn clifford(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let algebra = parse_macro_input!(input as Algebra);
    algebra.define().into()
}
