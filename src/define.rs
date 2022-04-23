use super::*;
use proc_macro2::{Ident, Span, TokenStream};
use quote::{quote, ToTokens};

// TODO mul blades

impl Algebra {
    pub fn define(&self) -> TokenStream {
        let traits = traits();
        let zero = zero();

        let blades = self
            .blades()
            .filter(|b| !b.0.is_empty())
            .map(|b| b.define());

        let grades = self.grades().map(|g| g.define());

        let even = SubAlgebra::Even(*self).define();
        let odd = SubAlgebra::Odd(*self).define();

        quote! {
            #traits

            #zero

            #(
                #blades
            )*

            #(
                #grades
            )*

            #even

            #odd
        }
    }
}

fn zero() -> TokenStream {
    quote! {
        #[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
        pub struct Zero;

        impl<T> std::ops::Add<T> for Zero {
            type Output = T;
            fn add(self, rhs: T) -> T {
                rhs
            }
        }

        impl<T: std::ops::Neg<Output = T>> std::ops::Sub<T> for Zero {
            type Output = T;
            fn sub(self, rhs: T) -> T {
                -rhs
            }
        }

        impl<T> std::ops::Mul<T> for Zero {
            type Output = Zero;
            fn mul(self, _rhs: T) -> Zero {
                Zero
            }
        }
    }
}

fn traits() -> TokenStream {
    quote! {
        pub trait Dot<Rhs> {
            type Output;
            fn dot(self, rhs: Rhs) -> Self::Output;
        }

        pub trait Commutator<Rhs> {
            type Output;
            fn commutator(self, rhs: Rhs) -> Self::Output;
        }

        pub trait Wedge<Rhs> {
            type Output;
            fn wedge(self, rhs: Rhs) -> Self::Output;
        }
    }
}

impl ToTokens for Multiplier {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let t = match self {
            Multiplier::One => quote! { f64 },
            Multiplier::Zero => quote! { Zero },
            Multiplier::NegOne => quote! { f64 },
        };
        tokens.extend(t);
    }
}

impl Grade {
    pub fn define(&self) -> TokenStream {
        let ty = self.type_ident();

        let struct_fields = self.blades().map(|b| {
            let f = b.field();
            let ty = b.type_ident();
            quote! { pub #f: #ty, }
        });

        let new_fn_fields = self.blades().map(|b| {
            let f = b.field();
            quote! { #f: f64 }
        });

        let new_fn_struct_fields = self.blades().map(|b| {
            let f = b.field();
            let ty = b.type_ident();
            quote! { #f: #ty(#f), }
        });

        let add_self_fields = self.blades().map(|b| {
            let f = b.field();
            quote! { #f: self.#f + rhs.#f, }
        });

        quote! {
            #[derive(Debug, Default, Copy, Clone, PartialEq)]
            pub struct #ty {
                #(#struct_fields)*
            }

            impl #ty {
                pub fn new(
                    #(#new_fn_fields,)*
                ) -> Self {
                    Self {
                        #(#new_fn_struct_fields)*
                    }
                }
            }

            impl std::ops::Add for #ty {
                type Output = Self;
                fn add(self, rhs: Self) -> Self {
                    Self {
                        #(#add_self_fields)*
                    }
                }
            }

            impl std::ops::Mul<Zero> for #ty {
                type Output = Zero;
                fn mul(self, rhs: Zero) -> Zero {
                    rhs
                }
            }

            impl std::ops::Add<Zero> for #ty {
                type Output = Self;
                fn add(self, _rhs: Zero) -> Self {
                    self
                }
            }

            impl std::ops::Sub<Zero> for #ty {
                type Output = Self;
                fn sub(self, _rhs: Zero) -> Self {
                    self
                }
            }
        }
    }

    pub fn type_ident(&self) -> Ident {
        let str = match self.0 {
            0 => "f64",
            1 => "Vector",
            2 => "Bivector",
            3 => "Trivector",
            4 => "Quadvector",
            5 => "Pentavector",
            6 => "Hexavector",
            7 => "Heptavector",
            8 => "Octovector",
            9 => "Nonavector",
            _ => unimplemented!("not implemented for grade: {}", self.0),
        };
        Ident::new(str, Span::mixed_site())
    }
}

impl Blade {
    pub fn define(&self) -> proc_macro2::TokenStream {
        let ty = self.type_ident();

        let square = (*self * *self).0;
        let square_expr = match square {
            Multiplier::One => quote! { self.0 * rhs.0 },
            Multiplier::Zero => quote! { Zero },
            Multiplier::NegOne => quote! { -(self.0 * rhs.0) },
        };

        quote::quote! {
            #[derive(Debug, Default, Copy, Clone, PartialEq, PartialOrd)]
            pub struct #ty(f64);

            impl std::ops::Mul<f64> for #ty {
                type Output = #ty;
                fn mul(self, rhs: f64) -> #ty {
                    #ty(self.0 * rhs)
                }
            }

            impl std::ops::Mul<#ty> for f64 {
                type Output = #ty;
                fn mul(self, rhs: #ty) -> #ty {
                    #ty(self * rhs.0)
                }
            }

            impl std::ops::Mul for #ty {
                type Output = #square;
                fn mul(self, rhs: Self) -> #square {
                    #square_expr
                }
            }

            impl std::ops::Mul<Zero> for #ty {
                type Output = Zero;
                fn mul(self, rhs: Zero) -> Zero {
                    rhs
                }
            }

            impl std::ops::Add<Zero> for #ty {
                type Output = #ty;
                fn add(self, _rhs: Zero) -> #ty {
                    self
                }
            }

            impl std::ops::Sub<Zero> for #ty {
                type Output = #ty;
                fn sub(self, _rhs: Zero) -> #ty {
                    self
                }
            }

            impl std::ops::Add for #ty {
                type Output = #ty;
                fn add(self, rhs: #ty) -> #ty {
                    #ty(self.0 + rhs.0)
                }
            }

            impl std::ops::Sub for #ty {
                type Output = #ty;
                fn sub(self, rhs: #ty) -> #ty {
                    #ty(self.0 - rhs.0)
                }
            }
        }
    }

    pub fn field(&self) -> Ident {
        if self.0.is_empty() {
            Ident::new("scalar", Span::mixed_site())
        } else {
            let mut output = "e".to_string();

            for i in 1..=self.1.sum() {
                if self.0.contains(i) {
                    output.push_str(&i.to_string());
                }
            }

            Ident::new(&output, Span::mixed_site())
        }
    }

    pub fn type_ident(&self) -> Ident {
        if self.0.is_empty() {
            Ident::new("f64", Span::mixed_site())
        } else {
            let mut output = "E".to_string();

            for basis in self.1.bases() {
                if self.0.contains(basis.0) {
                    output.push_str(&basis.0.to_string());
                }
            }

            Ident::new(&output, Span::mixed_site())
        }
    }
}

impl Multiplier {
    fn sign(&self) -> TokenStream {
        match self {
            Multiplier::NegOne => quote! { - },
            Multiplier::One => quote! { + },
            _ => quote! {},
        }
    }
}

impl SubAlgebra {
    pub fn define(&self) -> TokenStream {
        let ty = match self {
            SubAlgebra::Even(_) => quote! { Even },
            SubAlgebra::Odd(_) => quote! { Odd },
        };

        let blades = self.blades().map(|b| {
            let f = b.field();
            let ty = b.type_ident();
            quote! { pub #f: #ty, }
        });

        let add_self_fields = self.blades().map(|b| {
            let f = b.field();
            quote! { #f: self.#f + rhs.#f, }
        });

        let sub_self_fields = self.blades().map(|b| {
            let f = b.field();
            quote! { #f: self.#f - rhs.#f, }
        });

        let mul_self = MulProduct {
            lhs: *self,
            rhs: *self,
            f: |l: Blade, r: Blade| l * r,
        };

        let mul_self_output_ty = mul_self.output().type_ident();
        let mul_self_expr = mul_self.expr();

        let mul_opposite = MulProduct {
            lhs: *self,
            rhs: self.opposite(),
            f: |l, r| l * r,
        };
        let opposite = self.opposite().type_ident();
        let mul_opposite_output = mul_opposite.output().type_ident();
        let mul_opposite_expr = mul_opposite.expr();

        let mul_grades = self.algebra().grades().map(|grade| {
            let mul_grade = MulProduct {
                lhs: *self,
                rhs: grade,
                f: |l: Blade, r: Blade| l * r,
            };
            let grade = grade.type_ident();
            let output = mul_grade.output().type_ident();
            let expr = mul_grade.expr();
            quote! {
                impl std::ops::Mul<#grade> for #ty {
                    type Output = #output;
                    fn mul(self, rhs: #grade) -> Self::Output {
                        #expr
                    }
                }
            }
        });

        quote! {
            #[derive(Debug, Default, Copy, Clone, PartialEq)]
            pub struct #ty {
                #( #blades )*
            }

            impl std::ops::Add<Zero> for #ty {
                type Output = #ty;
                fn add(self, _rhs: Zero) -> Self::Output {
                    self
                }
            }

            impl std::ops::Sub<Zero> for #ty {
                type Output = #ty;
                fn sub(self, _rhs: Zero) -> Self::Output {
                    self
                }
            }

            impl std::ops::Mul<Zero> for #ty {
                type Output = Zero;
                fn mul(self, rhs: Zero) -> Self::Output {
                    rhs
                }
            }

            impl std::ops::Add for #ty {
                type Output = #ty;
                fn add(self, rhs: Self) -> Self {
                    Self {
                        #( #add_self_fields )*
                    }
                }
            }

            impl std::ops::Sub for #ty {
                type Output = #ty;
                fn sub(self, rhs: Self) -> Self {
                    Self {
                        #( #sub_self_fields )*
                    }
                }
            }

            impl std::ops::Mul for #ty {
                type Output = #mul_self_output_ty;
                fn mul(self, rhs: Self) -> Self::Output {
                    #mul_self_expr
                }
            }

            impl std::ops::Mul<#opposite> for #ty {
                type Output = #mul_opposite_output;
                fn mul(self, rhs: #opposite) -> Self::Output {
                    #mul_opposite_expr
                }
            }

            #( #mul_grades )*
        }
    }

    fn type_ident(&self) -> Ident {
        let str = match self {
            SubAlgebra::Even(_) => "Even",
            SubAlgebra::Odd(_) => "Odd",
        };
        Ident::new(str, Span::mixed_site())
    }
}

impl Type {
    pub fn type_ident(&self) -> Ident {
        match self {
            Type::Zero => Ident::new("Zero", Span::mixed_site()),
            Type::Grade(grade) => grade.type_ident(),
            Type::SubAlgebra(sub) => sub.type_ident(),
        }
    }
}

pub trait CompoundType {
    fn blades(&self) -> Vec<Blade>;
    fn type_ident(&self) -> proc_macro2::Ident;
}

impl CompoundType for Grade {
    fn blades(&self) -> Vec<Blade> {
        self.blades().collect()
    }

    fn type_ident(&self) -> Ident {
        self.type_ident()
    }
}

impl CompoundType for SubAlgebra {
    fn blades(&self) -> Vec<Blade> {
        self.blades().collect()
    }

    fn type_ident(&self) -> Ident {
        self.type_ident()
    }
}

pub struct MulProduct<A, B, F> {
    pub lhs: A,
    pub rhs: B,
    pub f: F,
}

impl<A, B, F> MulProduct<A, B, F>
where
    A: CompoundType,
    B: CompoundType,
    F: Fn(Blade, Blade) -> (Multiplier, Blade) + Copy,
{
    fn output(&self) -> Type {
        let f = self.f;
        self.lhs
            .blades()
            .into_iter()
            .flat_map(|lhs_b| {
                self.rhs
                    .blades()
                    .into_iter()
                    .map(move |rhs_b| f(lhs_b, rhs_b))
            })
            .filter_map(|(multiplier, blade)| {
                if matches!(multiplier, Multiplier::Zero) {
                    None
                } else {
                    Some(blade)
                }
            })
            .collect::<Type>()
    }

    fn expr(&self) -> TokenStream {
        let (ty, blades) = match self.output() {
            Type::Zero => return quote! { Zero },
            Type::Grade(grade) => {
                let ty = grade.type_ident();
                let blades = grade.blades().collect::<Vec<_>>();
                (ty, blades)
            }
            Type::SubAlgebra(sub) => {
                let ty = sub.type_ident();
                let blades = sub.blades().collect::<Vec<_>>();
                (ty, blades)
            }
        };

        let fields = blades.into_iter().map(|b| {
            let f = b.field();

            let products = self
                .lhs
                .blades()
                .into_iter()
                .flat_map(|lhs_b| {
                    self.rhs.blades().into_iter().map(move |rhs_b| {
                        let (product, blade) = lhs_b * rhs_b;
                        (lhs_b, rhs_b, product, blade)
                    })
                })
                .filter(|(_, _, multiplier, blade)| *multiplier != Multiplier::Zero && *blade == b)
                .map(|(lhs_b, rhs_b, multiplier, _)| {
                    let lhs_f = lhs_b.field();
                    let rhs_f = rhs_b.field();
                    (multiplier, quote! { self.#lhs_f * rhs.#rhs_f })
                })
                .collect::<Vec<_>>();

            if products.is_empty() {
                quote! { #f: Default::default(), }
            } else {
                let mut products = products.into_iter();
                let (mult, expr) = products.next().unwrap();
                let first = match mult {
                    Multiplier::NegOne => quote! { -(#expr) },
                    _ => quote! { #expr },
                };
                let rest = products.map(|(mult, expr)| {
                    let sign = mult.sign();
                    quote! { #sign #expr }
                });
                quote! { #f: #first #(#rest)* , }
            }
        });

        quote! {
            #ty {
                #( #fields )*
            }
        }
    }
}
