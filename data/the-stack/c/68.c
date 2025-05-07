/*********************************************************************************************************************/
/*   File Name: ang/base/text.h                                                                                      */
/*   Author: Ing. Jesus Rocha <chuyangel.rm@gmail.com>, July 2016.                                                   */
/*                                                                                                                   */
/*   Copyright (C) angsys, Jesus Angel Rocha Morales                                                                 */
/*   You may opt to use, copy, modify, merge, publish and/or distribute copies of the Software, and permit persons   */
/*   to whom the Software is furnished to do so.                                                                     */
/*   This software is distributed on an "AS IS" basis, WITHOUT WARRANTY OF ANY KIND, either express or implied.      */
/*                                                                                                                   */
/*********************************************************************************************************************/

#ifndef __ANG_BASE_H__
#error ang/base/base.h is not included
#elif !defined __ANG_BASE_STR_VIEW_H__
#define __ANG_BASE_TEXT_H__

#define MY_LINKAGE LINK
#define MY_CHAR_TYPE ang::text::char_type_by_encoding<MY_ENCODING>::char_type

#define MY_ENCODING ang::text::encoding::ascii
#include <ang/base/inline/str_view.hpp>
#undef MY_ENCODING

#define MY_ENCODING ang::text::encoding::unicode
#include <ang/base/inline/str_view.hpp>
#undef MY_ENCODING

#define MY_ENCODING ang::text::encoding::utf8
#include <ang/base/inline/str_view.hpp>
#undef MY_ENCODING

#define MY_ENCODING ang::text::encoding::utf16
#include <ang/base/inline/str_view.hpp>
#undef MY_ENCODING

#define MY_ENCODING ang::text::encoding::utf16_se
#include <ang/base/inline/str_view.hpp>
#undef MY_ENCODING

#define MY_ENCODING ang::text::encoding::utf16_le
#include <ang/base/inline/str_view.hpp>
#undef MY_ENCODING

#define MY_ENCODING ang::text::encoding::utf16_be
#include <ang/base/inline/str_view.hpp>
#undef MY_ENCODING

#define MY_ENCODING ang::text::encoding::utf32
#include <ang/base/inline/str_view.hpp>
#undef MY_ENCODING

#define MY_ENCODING ang::text::encoding::utf32_se
#include <ang/base/inline/str_view.hpp>
#undef MY_ENCODING

#define MY_ENCODING ang::text::encoding::utf32_le
#include <ang/base/inline/str_view.hpp>
#undef MY_ENCODING

#define MY_ENCODING ang::text::encoding::utf32_be
#include <ang/base/inline/str_view.hpp>
#undef MY_ENCODING

#undef MY_LINKAGE

namespace ang
{
	namespace text
	{

		template<> struct LINK str_view<void, text::encoding::auto_detect> {
			str_view();
			str_view(ang::nullptr_t const&);
			str_view(void* v, wsize s, text::encoding e);
			str_view(raw_str const& str);
			template<typename T, text::encoding E> inline str_view(str_view<T, E> str)
				: str_view(str.str(), str.size() * sizeof(typename text::char_type_by_encoding<E>::char_type), E) {
			}

			bool is_empty()const;
			void* ptr()const;
			wsize size()const;
			wsize count()const;
			wsize char_size()const;
			text::encoding encoding()const;

			template<text::encoding E> inline operator str_view<typename text::char_type_by_encoding<E>::char_type, E>() {
				return E == m_encoding ? str_view<typename text::char_type_by_encoding<E>::char_type, E>(
					(typename text::char_type_by_encoding<E>::str_type)m_value,
					m_size / sizeof(typename text::char_type_by_encoding<E>::char_type))
					: str_view<typename text::char_type_by_encoding<E>::char_type, E>();
			}
			template<text::encoding E> inline operator cstr_view<typename text::char_type_by_encoding<E>::char_type, E>()const {
				return E == m_encoding ? cstr_view<typename text::char_type_by_encoding<E>::char_type, E>(
					(typename text::char_type_by_encoding<E>::cstr_type)m_value,
					m_size / sizeof(typename text::char_type_by_encoding<E>::char_type))
					: cstr_view<typename text::char_type_by_encoding<E>::char_type, E>();
			}
			template<text::encoding E> inline str_view<typename text::char_type_by_encoding<E>::char_type, E> str() {
				return E == m_encoding ? str_view<typename text::char_type_by_encoding<E>::char_type, E>(
					(typename text::char_type_by_encoding<E>::str_type)m_value,
					m_size / sizeof(typename text::char_type_by_encoding<E>::char_type))
					: str_view<typename text::char_type_by_encoding<E>::char_type, E>();
			}
			template<text::encoding E> inline cstr_view<typename text::char_type_by_encoding<E>::char_type, E> cstr()const {
				return E == m_encoding ? cstr_view<typename text::char_type_by_encoding<E>::char_type, E>(
					(typename text::char_type_by_encoding<E>::cstr_type)m_value,
					m_size / sizeof(typename text::char_type_by_encoding<E>::char_type))
					: cstr_view<typename text::char_type_by_encoding<E>::char_type, E>();
			}

		private:
			void* m_value;
			wsize m_size;
			text::encoding m_encoding;
		};

		template<> struct LINK str_view<const void, text::encoding::auto_detect> {
			str_view();
			str_view(ang::nullptr_t const&);
			str_view(void const* v, wsize s, text::encoding e);
			str_view(raw_cstr const& str);
			str_view(raw_str const& str);
			template<typename T, text::encoding E> inline str_view(str_view<T, E> str)
				: str_view(str.cstr(), str.size() * sizeof(typename text::char_type_by_encoding<E>::char_type), E) {
			}
			template<typename T, text::encoding E> inline str_view(cstr_view<T, E> str)
				: str_view(str.cstr(), str.size() * sizeof(typename text::char_type_by_encoding<E>::char_type), E) {
			}
			template<typename T, wsize N> inline str_view(const T(&str)[N])
				: str_view(str, (N - 1) * sizeof(T), text::encoding_by_char_type<T>::value) {
			}

			bool is_empty()const;
			void const* ptr()const;
			wsize size()const;
			wsize count()const;
			wsize char_size()const;
			text::encoding encoding()const;

			template<text::encoding E> inline operator cstr_view<typename text::char_type_by_encoding<E>::char_type, E>()const {
				return E == m_encoding ? cstr_view<typename text::char_type_by_encoding<E>::char_type, E>(
					(typename text::char_type_by_encoding<E>::cstr_type)m_value,
					m_size / sizeof(typename text::char_type_by_encoding<E>::char_type))
					: cstr_view<typename text::char_type_by_encoding<E>::char_type, E>();
			}

			template<text::encoding E> inline cstr_view<typename text::char_type_by_encoding<E>::char_type, E> cstr()const {
				return E == m_encoding ? cstr_view<typename text::char_type_by_encoding<E>::char_type, E>(
					(typename text::char_type_by_encoding<E>::cstr_type)m_value,
					m_size / sizeof(typename text::char_type_by_encoding<E>::char_type))
					: cstr_view<typename text::char_type_by_encoding<E>::char_type, E>();
			}

		private:
			void const* m_value;
			wsize m_size;
			text::encoding m_encoding;
		};


		template<typename cstr1_t, typename cstr2_t> struct str_view_compare_helper;

		template<typename T1, text::encoding E1, typename T2, text::encoding E2>
		struct str_view_compare_helper<str_view<T1, E1>, str_view<T2, E2>>
		{
			inline static int compare(const str_view<T1, E1>& value1, const str_view<T2, E2>& value2) {
				if constexpr(E1 == text::encoding::auto_detect || E2 == text::encoding::auto_detect)
					return text::text_encoder<text::encoding::auto_detect>::compare(value1, value2);
				else
					return text::text_encoder<E1>::compare(value1.cstr(), value2);
			}
		};

		template<typename T1, text::encoding E1, typename T2>
		struct str_view_compare_helper<str_view<T1, E1>, const T2*>
		{
			inline static int compare(const str_view<T1, E1>& value1, const T2* value2) {
				if constexpr (E1 == text::encoding::auto_detect)
					return text::text_encoder<text::encoding::auto_detect>::compare(value1, str_view<const T2>(value2, 1)); //optimization hack: no real size needed for null termination string
				else
					return text::text_encoder<E1>::compare(value1.cstr(), value2);
			}
		};

		template<typename T1, typename T2, text::encoding E2>
		struct str_view_compare_helper<const T1*, str_view<T2, E2>>
		{
			inline static int compare(const T1* value1, const str_view<T2, E2>& value2) {
				if constexpr (E2 == text::encoding::auto_detect)
					return text::text_encoder<text::encoding::auto_detect>::compare(str_view<const T1>(value1, 1), value2); //optimization hack: no real size needed for null termination string
				else
					return text::text_encoder<text::encoding_by_char_type<T1>::value>::compare(value1, value2);
			}
		};

		template<typename T1, text::encoding E1, typename T2, wsize N2>
		struct str_view_compare_helper<str_view<T1, E1>, T2[N2]>
		{
			inline static int compare(const str_view<T1, E1>& value1, T2(&value2)[N2]) {
				if constexpr (E1 == text::encoding::auto_detect)
					return text::text_encoder<text::encoding::auto_detect>::compare(value1, str_view<T2>(value2, N2 - 1));
				else
					return text::text_encoder<E1>::compare(value1.cstr(), (T2*)value2);
			}
		};

		template<typename T1, wsize N1, typename T2, text::encoding E2>
		struct str_view_compare_helper<T1[N1], str_view<T2, E2>>
		{
			inline static int compare(T1(&value1)[N1], const str_view<T2, E2>& value2) {
				if constexpr (E2 == text::encoding::auto_detect)
					return text::text_encoder<text::encoding::auto_detect>::compare(str_view<T1>(value1, N1 - 1), value2);
				else
					return text::text_encoder<text::encoding_by_char_type<T1>::value>::compare((T1*)value1, value2);
			}
		};

		
		template<typename T, text::encoding E>
		struct str_view_compare_helper<str_view<T, E>, nullptr_t>
		{
			static inline int compare(const str_view<T, E>& value, nullptr_t const&) {
				if constexpr (E == text::encoding::auto_detect)
					return value.ptr() ? 1 : 0;
				else
					return value.cstr() ? 1 : 0;
			}
		};

		template<typename T, text::encoding E>
		struct str_view_compare_helper<nullptr_t, str_view<T, E>>
		{
			static inline int compare(nullptr_t const&, const str_view<T, E>& value) {
				if constexpr (E == text::encoding::auto_detect)
					return value.ptr() ? -1 : 0;
				else
					return value.cstr() ? -1 : 0;
			}
		};

		template<typename T1, text::encoding E1, typename T2, text::encoding E2>
		bool operator == (const str_view<T1, E1>& value1, const str_view<T2, E2>& value2) {
			return str_view_compare_helper<str_view<T1, E1>, str_view<T2, E2>>::compare(value1, value2) == 0;
		}
		template<typename T, text::encoding E, typename cstr_t>
		bool operator == (const str_view<T, E>& value1, cstr_t value2) {
			return str_view_compare_helper<str_view<T, E>, cstr_t>::compare(value1, value2) == 0;
		}
		template<typename T, text::encoding E, typename cstr_t>
		bool operator == (cstr_t value1, const str_view<T, E>& value2) {
			return str_view_compare_helper<cstr_t, str_view<T, E>>::compare(value1, value2) == 0;
		}


		template<typename T1, text::encoding E1, typename T2, text::encoding E2>
		bool operator != (const str_view<T1, E1>& value1, const str_view<T2, E2>& value2) {
			return str_view_compare_helper<str_view<T1, E1>, str_view<T2, E2>>::compare(value1, value2) != 0;
		}
		template<typename T, text::encoding E, typename cstr_t>
		bool operator != (const str_view<T, E>& value1, cstr_t value2) {
			return str_view_compare_helper<str_view<T, E>, cstr_t>::compare(value1, value2) != 0;
		}
		template<typename T, text::encoding E, typename cstr_t>
		bool operator != (cstr_t value1, const str_view<T, E>& value2) {
			return str_view_compare_helper<cstr_t, str_view<T, E>>::compare(value1, value2) != 0;
		}

		template<typename T1, text::encoding E1, typename T2, text::encoding E2>
		bool operator >= (const str_view<T1, E1>& value1, const str_view<T2, E2>& value2) {
			return str_view_compare_helper<str_view<T1, E1>, str_view<T2, E2>>::compare(value1, value2) >= 0;
		}
		template<typename T, text::encoding E, typename cstr_t>
		bool operator >= (const str_view<T, E>& value1, cstr_t value2) {
			return str_view_compare_helper<str_view<T, E>, cstr_t>::compare(value1, value2) >= 0;
		}
		template<typename T, text::encoding E, typename cstr_t>
		bool operator >= (cstr_t value1, const str_view<T, E>& value2) {
			return str_view_compare_helper<cstr_t, str_view<T, E>>::compare(value1, value2) >= 0;
		}

		template<typename T1, text::encoding E1, typename T2, text::encoding E2>
		bool operator <= (const str_view<T1, E1>& value1, const str_view<T2, E2>& value2) {
			return str_view_compare_helper<str_view<T1, E1>, str_view<T2, E2>>::compare(value1, value2) <= 0;
		}
		template<typename T, text::encoding E, typename cstr_t>
		bool operator <= (const str_view<T, E>& value1, cstr_t value2) {
			return str_view_compare_helper<str_view<T, E>, cstr_t>::compare(value1, value2) <= 0;
		}
		template<typename T, text::encoding E, typename cstr_t>
		bool operator <= (cstr_t value1, const str_view<T, E>& value2) {
			return str_view_compare_helper<cstr_t, str_view<T, E>>::compare(value1, value2) <= 0;
		}

		template<typename T1, text::encoding E1, typename T2, text::encoding E2>
		bool operator > (const str_view<T1, E1>& value1, const str_view<T2, E2>& value2) {
			return str_view_compare_helper<str_view<T1, E1>, str_view<T2, E2>>::compare(value1, value2) > 0;
		}
		template<typename T, text::encoding E, typename cstr_t>
		bool operator > (const str_view<T, E>& value1, cstr_t value2) {
			return str_view_compare_helper<str_view<T, E>, cstr_t>::compare(value1, value2) > 0;
		}
		template<typename T, text::encoding E, typename cstr_t>
		bool operator > (cstr_t value1, const str_view<T, E>& value2) {
			return str_view_compare_helper<cstr_t, str_view<T, E>>::compare(value1, value2) > 0;
		}

		template<typename T1, text::encoding E1, typename T2, text::encoding E2>
		bool operator < (const str_view<T1, E1>& value1, const str_view<T2, E2>& value2) {
			return str_view_compare_helper<str_view<T1, E1>, str_view<T2, E2>>::compare(value1, value2) < 0;
		}
		template<typename T, text::encoding E, typename cstr_t>
		bool operator < (const str_view<T, E>& value1, cstr_t value2) {
			return str_view_compare_helper<str_view<T, E>, cstr_t>::compare(value1, value2) < 0;
		}
		template<typename T, text::encoding E, typename cstr_t>
		bool operator < (cstr_t value1, const str_view<T, E>& value2) {
			return str_view_compare_helper<cstr_t, str_view<T, E>>::compare(value1, value2) < 0;
		}

	}

	inline str_view<const char> operator "" _sv(const char* str, wsize sz) { return str_view<const char>(str, sz); }
	inline str_view<const mchar> operator "" _svm(const char* str, wsize sz) { return str_view<const mchar>((mchar const*)str, sz); }
	inline str_view<const wchar_t> operator "" _sv(const wchar_t* str, wsize sz) { return str_view<const wchar_t>(str, sz); }
	inline str_view<const char16_t> operator "" _sv(const char16_t* str, wsize sz) { return str_view<const char16_t>(str, sz); }
	inline str_view<const char32_t> operator "" _sv(const char32_t* str, wsize sz) { return str_view<const char32_t>(str, sz); }

	inline raw_cstr_t operator "" _r(const char* str, wsize sz) { return str_view<const char>(str, sz); }
	inline raw_cstr_t operator "" _r(const wchar_t* str, wsize sz) { return str_view<const wchar_t>(str, sz); }
	inline raw_cstr_t operator "" _r(const char16_t* str, wsize sz) { return str_view<const char16_t>(str, sz); }
	inline raw_cstr_t operator "" _r(const char32_t* str, wsize sz) { return str_view<const char32_t>(str, sz); }

	namespace algorithms
	{
		template<typename T, text::encoding E>
		struct hash<str_view<T, E>> {
			static ulong64 make(str_view<T, E> const& value) {
				ulong64 h = 75025;
				windex i = 0, c = value.size();
				for (char32_t n = text::to_char32<false, text::is_endian_swapped<E>::value>(value.cstr(), i);
					n != 0;
					n = text::to_char32<false, text::is_endian_swapped<E>::value>(value.cstr(), i))
				{
					h = (h << 5) + h + n + 1;
				}
				return h;
			}
			ulong64 operator()(str_view<T, E> const& value)const {
				return make(value);
			}
		};

		template<>
		struct hash<cstr_t> {
			static ulong64 make(cstr_t const& value) {
				ulong64 h = 75025;
				windex i = 0, c = value.size();
				for (char32_t n = text::encoder::to_char32(value, i);
					n != 0;
					n = text::encoder::to_char32(value, i))
				{
					h = (h << 5) + h + n + 1;
				}
				return h;
			}
			ulong64 operator()(cstr_t const& value)const {
				return make(value);
			}
		};
	}

	template<typename T, T VALUE>
	struct enum_to_string
	{
		static const str_view<const char> value;
	};

#define TO_STRING_TEMPLATE(_LINK, _VALUE) namespace ang { template<> struct _LINK enum_to_string<decltype(_VALUE), _VALUE>	{ static const str_view<const char> value; }; }
#define TO_STRING_TEMPLATE_IMPLEMENT(_ENUM, _VALUE) const ang::str_view<const char>  ang::enum_to_string<_ENUM, _ENUM::_VALUE>::value = ANG_UTILS_TO_STRING_OBJ(_VALUE);

}


TO_STRING_TEMPLATE(LINK, ang::text::encoding::binary);
TO_STRING_TEMPLATE(LINK, ang::text::encoding::ascii);
TO_STRING_TEMPLATE(LINK, ang::text::encoding::unicode);
TO_STRING_TEMPLATE(LINK, ang::text::encoding::utf8);
TO_STRING_TEMPLATE(LINK, ang::text::encoding::utf16);
TO_STRING_TEMPLATE(LINK, ang::text::encoding::utf16_se);
TO_STRING_TEMPLATE(LINK, ang::text::encoding::utf16_le);
TO_STRING_TEMPLATE(LINK, ang::text::encoding::utf16_be);
TO_STRING_TEMPLATE(LINK, ang::text::encoding::utf32);
TO_STRING_TEMPLATE(LINK, ang::text::encoding::utf32_se);
TO_STRING_TEMPLATE(LINK, ang::text::encoding::utf32_le);
TO_STRING_TEMPLATE(LINK, ang::text::encoding::utf32_be);
TO_STRING_TEMPLATE(LINK, ang::text::encoding::auto_detect);

#endif//__ANG_BASE_TEXT_H__
