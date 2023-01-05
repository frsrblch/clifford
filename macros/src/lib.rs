extern crate proc_macro;

use syn::{parse::Parse, parse_macro_input, token::Comma, LitInt};

use crate::algebra::{Algebra, Basis};

mod algebra;
mod code_gen;

#[proc_macro_error::proc_macro_error]
#[proc_macro]
#[allow(non_snake_case)]
pub fn algebra(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let algebra = parse_macro_input!(input as Algebra);
    let tokens = algebra.define();
    tokens.into()
}

#[proc_macro_error::proc_macro_error]
#[proc_macro]
#[allow(non_snake_case)]
pub fn algebra_slim(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let mut algebra = parse_macro_input!(input as Algebra);
    algebra.slim = true;
    let tokens = algebra.define();
    tokens.into()
}

impl Parse for Algebra {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        if let Ok(int_algebra) = IntAlgebra::parse(input) {
            return Ok(Algebra::from(int_algebra));
        }

        let mut bases = vec![];
        loop {
            let char = input.parse::<syn::Ident>().and_then(|ident| {
                let str = ident.to_string();
                if str.len() == 1 {
                    Ok(str.chars().next().unwrap())
                } else {
                    proc_macro_error::abort!(ident.span(), "bases must be a single character")
                }
            })?;
            let _caret = input.parse::<syn::token::Caret>()?;
            let _two = input.parse::<LitInt>().and_then(|int| {
                let u32 = int.base10_parse::<u32>()?;
                if u32 == 2 {
                    Ok(int)
                } else {
                    proc_macro_error::abort!(int.span(), "expected exponent 2")
                }
            })?;
            let _eq = input.parse::<syn::Token![==]>()?;
            let sqr = if input.peek(syn::Token![-]) {
                let _neg = input.parse::<syn::Token![-]>()?;
                let _one = input.parse::<LitInt>().and_then(|int| {
                    let u32 = int.base10_parse::<u32>()?;
                    if u32 == 1 {
                        Ok(int)
                    } else {
                        proc_macro_error::abort!(int.span(), "expected 1, 0, or -1")
                    }
                });
                algebra::Square::Neg
            } else {
                let int = input.parse::<LitInt>()?;
                let u32 = int.base10_parse::<u32>()?;
                match u32 {
                    1 => algebra::Square::Pos,
                    0 => algebra::Square::Zero,
                    _ => proc_macro_error::abort!(int.span(), "expected 1, 0, or -1"),
                }
            };

            let _: Comma = input.parse()?;

            bases.push(Basis { char, sqr });

            if input.is_empty() {
                let bases = bases.leak();
                return Ok(Algebra { bases, slim: false });
            }
        }
    }
}

pub(crate) struct IntAlgebra {
    pos: u32,
    neg: u32,
    zero: u32,
}

impl Parse for IntAlgebra {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let pos: u32 = input.parse::<LitInt>()?.base10_parse()?;

        if input.is_empty() {
            return Ok(IntAlgebra {
                pos,
                neg: 0,
                zero: 0,
            });
        }

        let _: Comma = input.parse()?;

        let neg: u32 = input.parse::<LitInt>()?.base10_parse()?;

        if input.is_empty() {
            return Ok(IntAlgebra { pos, neg, zero: 0 });
        }

        let _: Comma = input.parse()?;

        let zero = input.parse::<LitInt>()?.base10_parse()?;

        Ok(IntAlgebra { pos, neg, zero })
    }
}

impl From<IntAlgebra> for Algebra {
    fn from(a: IntAlgebra) -> Self {
        let mut bases = Vec::with_capacity((a.pos + a.neg + a.zero) as usize);
        for p in 0..a.pos {
            bases.push(Basis {
                char: std::char::from_digit(p + 1, 16).expect("std::char::from_digit out of range"),
                sqr: algebra::Square::Pos,
            });
        }
        for q in 0..a.neg {
            bases.push(Basis {
                char: std::char::from_digit(q + a.pos + 1, 16)
                    .expect("std::char::from_digit out of range"),
                sqr: algebra::Square::Neg,
            });
        }
        for z in 0..a.zero {
            bases.push(Basis {
                char: std::char::from_digit(z + a.pos + a.neg + 1, 16)
                    .expect("std::char::from_digit out of range"),
                sqr: algebra::Square::Zero,
            });
        }
        Algebra {
            bases: Box::leak(bases.into()),
            slim: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::algebra::Square;

    use super::*;

    #[test]
    fn parse_test() {
        let g3: Algebra = syn::parse_str("3").unwrap();
        assert_eq!(3, g3.bases.len());
        assert!(g3.bases.iter().all(|b| b.sqr == Square::Pos));

        let cga3: Algebra = syn::parse_str("4, 1").unwrap();
        assert_eq!(5, cga3.bases.len());
        assert_eq!(
            4,
            cga3.bases.iter().filter(|b| b.sqr == Square::Pos).count()
        );
        assert_eq!(
            1,
            cga3.bases.iter().filter(|b| b.sqr == Square::Neg).count()
        );

        let pga3: Algebra = syn::parse_str("3, 0, 1").unwrap();
        assert_eq!(4, pga3.bases.len());
        assert_eq!(
            3,
            pga3.bases.iter().filter(|b| b.sqr == Square::Pos).count()
        );
        assert_eq!(
            1,
            pga3.bases.iter().filter(|b| b.sqr == Square::Zero).count()
        );
    }
}
