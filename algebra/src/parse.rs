use crate::Square;
use proc_macro2::Span;
use syn::{
    braced,
    parse::Parse,
    punctuated::Punctuated,
    token::{Brace, Comma},
    Error, Ident, LitInt, Token,
};

pub struct Algebra {
    bases: Bases,
    comma: Option<Token![,]>,
    blades: Option<Blades>,
}

impl From<Algebra> for crate::Algebra {
    fn from(value: Algebra) -> Self {
        if let Some(blades) = value.blades {
            crate::Algebra::new_with_fields(value.bases.bases, blades.blades)
        } else {
            crate::Algebra::new(value.bases.bases)
        }
    }
}

impl Parse for Algebra {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let bases = input.parse()?;
        if input.is_empty() {
            return Ok(Algebra {
                bases,
                comma: None,
                blades: None,
            });
        }
        Ok(Algebra {
            bases,
            comma: input.parse().map(Some)?,
            blades: input.parse().map(Some)?,
        })
    }
}

pub struct Bases {
    bases_token: BasesToken,
    brace: Brace,
    bases: Punctuated<Basis, Token![,]>,
}

impl Parse for Bases {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let content;
        Ok(Self {
            bases_token: input.parse()?,
            brace: braced!(content in input),
            bases: content.parse_terminated(Parse::parse)?,
        })
    }
}

pub struct BasesToken(Span);

impl Parse for BasesToken {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let ident = input.parse::<Ident>()?;
        let string = ident.to_string();
        if string == "bases" {
            Ok(BasesToken(ident.span()))
        } else {
            Err(Error::new(
                ident.span(),
                format!("expected 'bases', found {string}"),
            ))
        }
    }
}

pub struct Basis {
    char: Char,
    caret: Token![^],
    two: Two,
    eq: Token![==],
    square: Square,
}

impl From<Basis> for crate::Basis {
    fn from(value: Basis) -> Self {
        let Basis { char, square, .. } = value;
        crate::Basis {
            char: char.char,
            square,
        }
    }
}

impl Parse for Basis {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        Ok(Basis {
            char: input.parse()?,
            caret: input.parse()?,
            two: input.parse()?,
            eq: input.parse()?,
            square: input.parse()?,
        })
    }
}

pub struct Char {
    span: Span,
    char: char,
}

impl Parse for Char {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let ident = input.parse::<Ident>()?;
        let string = ident.to_string();
        let mut chars = string.chars();
        if let (Some(char), None) = (chars.next(), chars.next()) {
            Ok(Char {
                span: ident.span(),
                char,
            })
        } else {
            Err(Error::new(
                ident.span(),
                format!("basis ident must be a single character: {string}"),
            ))
        }
    }
}

struct Two;

impl Parse for Two {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let lit_int = input.parse::<LitInt>()?;
        let int = lit_int.base10_parse::<i32>()?;
        if int == 2 {
            Ok(Two)
        } else {
            Err(Error::new(lit_int.span(), "expected 2"))
        }
    }
}

impl Parse for Square {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let lit_int = input.parse::<LitInt>()?;
        match lit_int.base10_parse::<i32>()? {
            -1 => Ok(Square::Neg),
            0 => Ok(Square::Zero),
            1 => Ok(Square::Pos),
            _ => Err(Error::new(lit_int.span(), "expected 1, 0, or -1")),
        }
    }
}

impl Parse for crate::Algebra {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        if let Ok(int_algebra) = IntAlgebra::parse(input) {
            return Ok(crate::Algebra::from(int_algebra));
        }

        if let Ok(algebra) = Algebra::parse(input) {
            return Ok(crate::Algebra::from(algebra));
        }

        Err(Error::new(input.span(), "unrecognized input, "))

        // let mut bases = vec![];
        // loop {
        //     let char = input.parse::<syn::Ident>().and_then(|ident| {
        //         let str = ident.to_string();
        //         if str.len() == 1 {
        //             Ok(str.chars().next().unwrap())
        //         } else {
        //             Err(Error::new(ident.span(), "bases must be a single character"))
        //         }
        //     })?;
        //     let _caret = input.parse::<syn::token::Caret>()?;
        //     let _two = input.parse::<LitInt>().and_then(|int| {
        //         let u32 = int.base10_parse::<u32>()?;
        //         if u32 == 2 {
        //             Ok(int)
        //         } else {
        //             Err(Error::new(int.span(), "expected 2"))
        //         }
        //     })?;
        //     let _eq = input.parse::<syn::Token![==]>()?;
        //     let lit_int = input.parse::<LitInt>()?;
        //     let sqr = match lit_int.base10_parse::<i32>()? {
        //         -1 => Square::Neg,
        //         0 => Square::Zero,
        //         1 => Square::Pos,
        //         _ => return Err(Error::new(lit_int.span(), "expected 1, 0, or -1")),
        //     };
        //     bases.push(crate::Basis { char, sqr });

        //     let _: Comma = input.parse()?;

        //     if input.is_empty() {
        //         return Ok(crate::Algebra::new(bases));
        //     }
        // }
    }
}

pub struct Blades {
    token: BladesToken,
    brace: Brace,
    blades: Punctuated<Ident, Token![,]>,
}

impl Parse for Blades {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let content;
        Ok(Self {
            token: input.parse()?,
            brace: braced!(content in input),
            blades: content.parse_terminated(Ident::parse)?,
        })
    }
}

struct BladesToken(Span);

impl Parse for BladesToken {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        const STR: &str = "blades";
        let ident = input.parse::<Ident>()?;
        if ident == STR {
            Ok(BladesToken(ident.span()))
        } else {
            Err(Error::new(ident.span(), format!("expected '{STR}'")))
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
        let pos = input.parse::<LitInt>()?.base10_parse()?;

        if input.is_empty() {
            return Ok(IntAlgebra {
                pos,
                neg: 0,
                zero: 0,
            });
        }

        let _: Comma = input.parse()?;

        let neg = input.parse::<LitInt>()?.base10_parse()?;

        if input.is_empty() {
            return Ok(IntAlgebra { pos, neg, zero: 0 });
        }

        let _: Comma = input.parse()?;

        let zero = input.parse::<LitInt>()?.base10_parse()?;

        Ok(IntAlgebra { pos, neg, zero })
    }
}

impl From<IntAlgebra> for crate::Algebra {
    fn from(a: IntAlgebra) -> Self {
        use Square::*;
        let IntAlgebra { pos, neg, zero } = a;
        let pos = (0..pos).map(|_| Pos);
        let neg = (0..neg).map(|_| Neg);
        let zero = (0..zero).map(|_| Zero);
        let bases = pos.chain(neg).chain(zero).enumerate().map(|(i, square)| {
            let char =
                std::char::from_digit(i as u32, 16).expect("std::char::from_digit out of range");
            square.basis(char)
        });
        crate::Algebra::new(bases)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_test() {
        let g3: crate::Algebra = syn::parse_str("3").unwrap();
        assert_eq!(3, g3.bases.len());
        assert!(g3.bases.iter().all(|b| b.square == Square::Pos));

        let cga3: crate::Algebra = syn::parse_str("4, 1").unwrap();
        assert_eq!(5, cga3.bases.len());
        assert_eq!(
            4,
            cga3.bases
                .iter()
                .filter(|b| b.square == Square::Pos)
                .count()
        );
        assert_eq!(
            1,
            cga3.bases
                .iter()
                .filter(|b| b.square == Square::Neg)
                .count()
        );

        let pga3: crate::Algebra = syn::parse_str("3, 0, 1").unwrap();
        assert_eq!(4, pga3.bases.len());
        assert_eq!(
            3,
            pga3.bases
                .iter()
                .filter(|b| b.square == Square::Pos)
                .count()
        );
        assert_eq!(
            1,
            pga3.bases
                .iter()
                .filter(|b| b.square == Square::Zero)
                .count()
        );

        let g2: crate::Algebra = syn::parse_str("bases { x ^ 2 == 1, y ^ 2 == 1 }").unwrap();
        assert_eq!(g2.bases[0], Square::Pos.basis('x'));
        assert_eq!(g2.bases[1], Square::Pos.basis('y'));

        let g2_rev: crate::Algebra = syn::parse_str(
            "\
            bases {
                x ^ 2 == 1,\
                y ^ 2 == 1,\
            },
            blades {
                yx
            }
            ",
        )
        .unwrap();
        assert!(g2_rev.ordering.is_flipped(crate::Blade(0b11)));
    }

    #[test]
    fn parse_neg_int() {
        let lit_int: syn::LitInt = syn::parse_str("-1").unwrap();
        let int = lit_int.base10_parse::<i32>().unwrap();
        assert_eq!(-1, int);
    }

    #[test]
    fn parse_basis() {
        let basis: Basis = syn::parse_str("x ^ 2 == -1").unwrap();
        assert_eq!(basis.char.char, 'x');
        assert_eq!(basis.square, Square::Neg);

        assert!(syn::parse_str::<Basis>("xy ^ 2 == 1").is_err());
        assert!(syn::parse_str::<Basis>("x * 2 == 1").is_err());
        assert!(syn::parse_str::<Basis>("x ^ 3 == 1").is_err());
        assert!(syn::parse_str::<Basis>("x ^ 2 != 1").is_err());
        assert!(syn::parse_str::<Basis>("x ^ 2 == 2").is_err());
    }

    #[test]
    fn parse_bases() {
        let bases: Bases = syn::parse_str("bases { x ^ 2 == 1, y ^ 2 == 0 }").unwrap();
        assert_eq!(2, bases.bases.len());
    }

    #[test]
    fn pga_3d_parse() {
        let algebra: crate::Algebra = syn::parse_str(
            "bases {
                x ^ 2 == 1,
                y ^ 2 == 1,
                z ^ 2 == 1,
                w ^ 2 == 0,
            },
            blades {
                xy, yz, zx, wz, wy, wx, zyx, wxy, wzx, wyz, xyzw
            }",
        )
        .unwrap();
        dbg!(&algebra);

        // let path = r#"C:\Users\Farseer\IdeaProjects\clifford\output.rs"#;
        // std::fs::write(path, algebra.define().to_string()).unwrap();
        // std::process::Command::new("rustfmt")
        //     .arg(path)
        //     .output()
        //     .unwrap();
    }

    #[test]
    fn cga_3d_parse() {
        let algebra: crate::Algebra = syn::parse_str(
            "bases {
                x ^ 2 == 1,
                y ^ 2 == 1,
                z ^ 2 == 1,
                e ^ 2 == 1,
                E ^ 2 == -1,
            },
            blades {
                xy, yz, zx, ez, ey, ex, zyx, exy, ezx, eyz, xyze,
                xE, yE, zE, eE, yzE, zxE, xyE, ezE, eyE, exE, zyxE, exyE, ezxE, eyzE, xyzeE,
            }",
        )
        .unwrap();
        dbg!(&algebra);
        // panic!("done");
    }
}
