#ifndef DICT_TEST_H
#define DICT_TEST_H

#include <unordered_map>
#include <string>
#include <vector>
#include <iterator>
#include "../Dict.h"
#include <algorithm>

using dict_t = HashDict<std::string, int>;
using C = dict_t;
using dict_value_t = dict_t::value_type;
using dict_key_t = dict_t::key_type;
using dict_mapped_t = dict_t::mapped_type;
using unord_map_t = std::unordered_map<std::string, int>;

template <class Map>
auto to_vector(const Map & m)
{
	using value_t = typename Map::value_type;
	std::vector<std::pair<dict_key_t, dict_mapped_t>> v;
	v.reserve(m.size());
	for(const auto & item: m)
	{
		v.push_back(std::make_pair(item.first, item.second));
	}
	std::sort(v.begin(), v.end());
	return v;
}
const std::vector<std::string>& default_keys(std::size_t = 0, std::size_t = 128);

std::vector<dict_key_t> get_keys(size_t begin = 0, size_t end = 128)
{
	assert(begin < end);
	assert(end <= default_keys().size());
	std::vector<std::string> keys(end - begin);
	std::copy(default_keys().begin() + begin, default_keys().begin() + end, keys.begin());
	return keys;
}

std::vector<dict_mapped_t> get_values(size_t begin = 0, size_t end = 128)
{
	std::vector<dict_mapped_t> values(end - begin);
	std::iota(values.begin(), values.end(), begin);
	return values;
}
std::vector<dict_value_t> get_kvps(size_t begin = 0, size_t end = 128)
{
	auto keys = get_keys(begin, end);
	auto values = get_values(begin, end);
	std::vector<dict_value_t> v;
	v.reserve(end - begin);
	for(size_t i = 0; i < end - begin; ++i)
	{
		v.emplace_back(keys[i], values[i]);
	}
	return v;
}
bool observably_same(const dict_t & d, const unord_map_t & m)
{
	bool same = true;
	if((d.size() != m.size()))
	{
		same = false;	
		std::cerr << "Sizes not equal." << std::endl;
	}
	auto v1 = to_vector(d);
	auto v2 = to_vector(m);
	if(d.size() != m.size())
	{
		same = false;
		std::cerr << "Vector sizes not equal." << std::endl;
	}
	if(v1 != v2)
	{
		same = false;
		auto [a, b] = std::mismatch(d.begin(), d.end(), m.begin(), m.end());

		std::cerr << "Vectors not equal!" << '\n' << 
			     "Mismatch found at " << std::distance(d.begin(), a) << '.' << '\n' << 
			     "Values: (" << a->first << ", " << a->second << ") \t (" << b->first << ", " << b->second << ")" << '\n' << std::endl;
	}
	return same;
}







const std::vector<std::string>& default_keys(size_t begin, size_t end)
{
	static const std::vector<std::string> keys{
		"zero",
		"one",
		"two",
		"three",
		"four",
		"five",
		"six",
		"seven",
		"eight",
		"nine",
		"ten",
		"eleven",
		"twelve",
		"thirteen",
		"fourteen",
		"fifteen",
		"sixteen",
		"seventeen",
		"eighteen",
		"nineteen",
		"twenty",
		"twenty-one",
		"twenty-two",
		"twenty-three",
		"twenty-four",
		"twenty-five",
		"twenty-six",
		"twenty-seven",
		"twenty-eight",
		"twenty-nine",
		"thirty",
		"thirty-one",
		"thirty-two",
		"thirty-three",
		"thirty-four",
		"thirty-five",
		"thirty-six",
		"thirty-seven",
		"thirty-eight",
		"thirty-nine",
		"forty",
		"forty-one",
		"forty-two",
		"forty-three",
		"forty-four",
		"forty-five",
		"forty-six",
		"forty-seven",
		"forty-eight",
		"forty-nine",
		"fifty",
		"fifty-one",
		"fifty-two",
		"fifty-three",
		"fifty-four",
		"fifty-five",
		"fifty-six",
		"fifty-seven",
		"fifty-eight",
		"fifty-nine",
		"sixty",
		"sixty-one",
		"sixty-two",
		"sixty-three",
		"sixty-four",
		"sixty-five",
		"sixty-six",
		"sixty-seven",
		"sixty-eight",
		"sixty-nine",
		"seventy",
		"seventy-one",
		"seventy-two",
		"seventy-three",
		"seventy-four",
		"seventy-five",
		"seventy-six",
		"seventy-seven",
		"seventy-eight",
		"seventy-nine",
		"eighty",
		"eighty-one",
		"eighty-two",
		"eighty-three",
		"eighty-four",
		"eighty-five",
		"eighty-six",
		"eighty-seven",
		"eighty-eight",
		"eighty-nine",
		"ninety",
		"ninety-one",
		"ninety-two",
		"ninety-three",
		"ninety-four",
		"ninety-five",
		"ninety-six",
		"ninety-seven",
		"ninety-eight",
		"ninety-nine",
		"one hundred",
		"one hundred and one",
		"one hundred and two",
		"one hundred and three",
		"one hundred and four",
		"one hundred and five",
		"one hundred and six",
		"one hundred and seven",
		"one hundred and eight",
		"one hundred and nine",
		"one hundred and ten",
		"one hundred and eleven",
		"one hundred and twelve",
		"one hundred and thirteen",
		"one hundred and fourteen",
		"one hundred and fifteen",
		"one hundred and sixteen",
		"one hundred and seventeen",
		"one hundred and eighteen",
		"one hundred and nineteen",
		"one hundred and twenty",
		"one hundred and twenty-one",
		"one hundred and twenty-two",
		"one hundred and twenty-three",
		"one hundred and twenty-four",
		"one hundred and twenty-five",
		"one hundred and twenty-six",
		"one hundred and twenty-seven"
	};
	return keys;
}


#endif /* DICT_TEST_H */
