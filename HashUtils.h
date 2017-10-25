#ifndef HASH_UTILS_H
#define HASH_UTILS_H
#include <climits>
#include <limits>
#include <utility>

namespace{
	
template <class SizeType, size_t ... I>
constexpr auto get_valid_hash_dict_sizes(std::index_sequence<I...>)
{
	return	std::array<SizeType, sizeof...(I)>{(SizeType(1) << I) ...};
}

} /* anonymous namespace */
namespace detail{
	// MUH PERF.
	constexpr inline void compiler_assume(bool condition) noexcept
	{
		#ifdef __GNUC__
			if(not condition)
				__builtin_unreachable();
		#else 
		  #ifdef __clang__
		    #if __has_builtin(__builtin_assume)
			__builtin_assume(condition);	
		    #elif __has_builtin(__builtin_unreachable)
			if(not condition)
				__builtin_unreachable();
		    #endif
		  #else
		    #ifdef __MSC_VER
			__assume(condition);
		    #endif
		  #endif
		#endif
	}
	template <bool PreCondition>
	constexpr inline void compiler_assume_if(bool condition) noexcept
	{
		if constexpr(PreCondition)
			compiler_assume(condition);
	}
} /* namespace detail */
template <class SizeType>
constexpr auto valid_hash_dict_sizes()
{
	return get_valid_hash_dict_sizes<SizeType>(std::make_index_sequence<sizeof(SizeType) * CHAR_BIT - 1>{});
}

template <class SizeType>
struct PowersOfTwoPolicy
{
	using array_t = std::array<SizeType, sizeof(SizeType) * CHAR_BIT - 1>;
public:
	template <bool NextLargest>
	constexpr SizeType next_size(const SizeType current_size) const
	{
		const auto pos = lower_bound_of(current_size);
		if constexpr(NextLargest)
		{
			const SizeType size_found = *pos;
			if((size_found > current_size) or (size_found == max_size()))
				return size_found;
			else
				return *(pos + 1);
		}
		else
		{
			const SizeType size_found = *pos;
			if((size_found < current_size) or (size_found == min_size()))
				return size_found;
			else
				return *(pos - 1);
		}
	}
	constexpr SizeType operator()(const SizeType requested_size, const SizeType count, const float max_load_factor) const
	{
		const SizeType mas = min_allowable_size(count, max_load_factor);
		const bool too_small = requested_size < mas;
		if(too_small)
		{
			return round_to_next_largest(mas);
		}
		return round_to_next_largest(requested_size);
	}
	constexpr SizeType operator()(const SizeType count, const float max_load_factor) const
	{
		const SizeType sz = min_allowable_size(count, max_load_factor);
		return round_to_next_largest(sz);
	}
	constexpr SizeType max_size() const
	{
		return *(valid_sizes.end() - 1);
	}
	constexpr SizeType min_size() const
	{
		return *(valid_sizes.begin());
	}
private:
	constexpr typename array_t::const_iterator lower_bound_of(const SizeType sz) const
	{
		return std::lower_bound(valid_sizes.cbegin(), valid_sizes.cend() - 1, sz);
	}
	constexpr SizeType round_to_next_largest(const SizeType sz) const
	{
		return *lower_bound_of(sz);
	}
	constexpr SizeType min_allowable_size(const SizeType count, const float max_load_factor) const
	{
		return std::ceil(count / max_load_factor);
	}
	static constexpr const array_t valid_sizes{valid_hash_dict_sizes<SizeType>()};
};
struct DoNothingOnRemoval
{
	template <class K, class V>
	void operator()(K& k, V& v) const
	{
	
	}

};

#endif /* HASH_UTILS_H */

