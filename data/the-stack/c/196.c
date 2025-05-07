#pragma once
#ifndef _RIBBON_REFSTRING_H_
#define _RIBBON_REFSTRING_H_
namespace Ribbon {

namespace RefStringInternal {

struct StringMemoryImage final
{
	std::atomic_uint		m_refCount;	// 4
	uint16_t				m_length;	// 2
	char16_t				m_str[1];	// 10(=4), 26(=12), 42(=20)

	uint32_t	DecrementReference();
	uint32_t	IncrementReference() { return ++m_refCount; }

	StringMemoryImage() = delete;
	StringMemoryImage(const StringMemoryImage&) = delete;
	StringMemoryImage(StringMemoryImage&&) = delete;
	StringMemoryImage& operator = (const StringMemoryImage&) = delete;
};

} // namespace RefStringInternal

class RefString final
{
private:
	static char16_t Length0Text[];
	RefStringInternal::StringMemoryImage* m_ptr;

public:
	RefString() : m_ptr(nullptr) {}

	// copy context
	RefString(const RefString& src) {
		m_ptr = src.m_ptr;
		if (m_ptr != nullptr) m_ptr->IncrementReference();
	}
	RefString& operator = (const RefString& src) {
		if (m_ptr != nullptr) m_ptr->DecrementReference();
		// do not touch m_ptr content after DecrementReference
		m_ptr = src.m_ptr;
		if (m_ptr != nullptr) m_ptr->IncrementReference();
		return *this;
	}
	// move context
	RefString(RefString&& src) {
		m_ptr = src.m_ptr;
		src.m_ptr = nullptr;
	}
	RefString& operator = (RefString&& src) {
		if (m_ptr != nullptr) m_ptr->DecrementReference();
		// do not touch m_ptr content after DecrementReference
		m_ptr = src.m_ptr;
		src.m_ptr = nullptr;
		return *this;
	}
	// destructor
	~RefString() {
		if (m_ptr != nullptr) m_ptr->DecrementReference();
		// do not touch m_ptr content after DecrementReference
	}

	// constructors
	RefString(const char16_t* src, size_t length);
	RefString(const char16_t* src) :
		RefString(src, textlen(src))
	{}
	RefString(const std::u16string& src) :
		RefString(src.c_str(), src.length())
	{}
	RefString& operator = (const char16_t* src) {
		return (*this = RefString(src));
	}
	RefString(const std::string& src)
		: RefString(to_utf16(src))
	{}
	RefString(const char* src)
		: RefString(to_utf16(src))
	{}
	RefString& operator = (const std::u16string& src) {
		return (*this = RefString(src));
	}

	// Accessor
	const std::string u8str() const {
		return to_utf8(m_ptr ? m_ptr->m_str : Length0Text);
	}
	const char16_t* u16ptr() const {
		return m_ptr ? m_ptr->m_str : Length0Text;
	}
	std::u16string u16str() const {
		return m_ptr ? std::u16string(m_ptr->m_str, m_ptr->m_length) : std::u16string();
	}
	size_t length() const {
		return m_ptr ? static_cast<size_t>(m_ptr->m_length) : 0;
	}
	char16_t& operator [] (size_t idx) {
		char16_t* ptr = m_ptr ? m_ptr->m_str : Length0Text;
		return ptr[idx];
	}
	const char16_t& operator [] (size_t idx) const {
		return u16ptr()[idx];
	}

	// Compare
	int Compare(const RefString& rhs) const {
		return (m_ptr == rhs.m_ptr) ? 0 :
			(m_ptr == nullptr) ? -1 :
			(rhs.m_ptr == nullptr) ? 1 :
			textcmp(m_ptr->m_str, rhs.m_ptr->m_str);
	}
	bool operator < (const RefString& rhs) const {
		return Compare(rhs) < 0;
	}
	bool operator > (const RefString& rhs) const {
		return Compare(rhs) > 0;
	}
	bool operator == (const RefString& rhs) const {
		return Compare(rhs) == 0;
	}
	operator bool () const {
		return m_ptr != nullptr;
	}

	// others
	void clear() {
		if (m_ptr != nullptr) m_ptr->DecrementReference();
		m_ptr = nullptr;
	}

	// for diagnotics
	uint32_t GetReferenceCount() { return m_ptr ? (uint32_t)m_ptr->m_refCount : 0; }
	static uint32_t GetActiveInstanceCount();
};

} // Ribbon
#endif // _RIBBON_REFSTRING_H_
