extern crate proc_macro;

mod algebra;
mod code_gen;

use algebra::Algebra;
use itertools::Itertools;
use proc_macro2::Ident;
use syn::parse::{Parse, ParseStream};
use syn::token::Comma;
use syn::LitInt;
use syn::{bracketed, parse_macro_input};

#[proc_macro]
pub fn clifford(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let algebra = parse_macro_input!(input as Algebra);
    algebra.define().into()
}

impl Parse for Algebra {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let one: LitInt = input.parse()?;
        let _comma: Comma = input.parse()?;
        let neg_one: LitInt = input.parse()?;
        let _comma: Comma = input.parse()?;
        let zero: LitInt = input.parse()?;

        let one = one.base10_parse()?;
        let neg_one = neg_one.base10_parse()?;
        let zero = zero.base10_parse()?;

        match input.parse::<Comma>() {
            Ok(_comma) => {
                let content;
                let _bracket = bracketed!(content in input);
                let idents = content.parse_terminated::<_, Comma>(Ident::parse)?;

                let idents = idents
                    .into_iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>();

                if !idents.iter().all_unique() {
                    panic!("all bases must be unique");
                }

                let chars = idents
                    .iter()
                    .map(|str| {
                        if str.chars().count() != 1 {
                            panic!("bases must be a single char: {}", str)
                        }
                        str.chars().next().unwrap()
                    })
                    .collect::<Vec<_>>();

                if chars.len() as u8 != one + neg_one + zero {
                    panic!(
                        "number of base identifiers must match bases: {} + {} + {} != {}",
                        one,
                        neg_one,
                        zero,
                        chars.len()
                    );
                }

                Ok(Algebra::new_bases(one, neg_one, zero, chars.leak()))
            }
            Err(_) => Ok(Algebra::new(one, neg_one, zero)),
        }
    }
}
