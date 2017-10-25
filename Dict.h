#ifndef DICT_H
#define DICT_H
#include <functional>
#include <algorithm>
#include <vector>
#include <type_traits>
#include <memory>
#include <limits>
#include <random>
#include <boost/iterator/iterator_facade.hpp>
#include "HashUtils.h"



#include <cassert>
#include <iostream>

template <class Key>
struct IdentityOrEquality
{
	IdentityOrEquality() = default;
	static constexpr const std::equal_to<Key> equal_to_{};
	bool operator()(const Key & left, const Key & right) const
	{
		if constexpr(std::is_arithmetic_v<Key> and (sizeof(Key*) < sizeof(Key))) 
			return  (std::addressof(left) == std::addressof(right)) or
				equal_to_(left, right);
		else
			return equal_to_(left, right);
	}

};

template <class Key>
struct KeyInfo
{
	KeyInfo(const Key & k):
		hash_(0), index_(std::numeric_limits<size_t>::max()), key_(&k)
	{
		
	}
	KeyInfo(const Key & k, std::size_t hash_v, std::size_t container_length):
		hash_(0), index_(std::numeric_limits<size_t>::max()), key_(&k)
	{
		
	}
	std::size_t hash_;
	std::size_t index_;
	const Key * key_;
	std::size_t hash() const { return hash_; }
	std::size_t index() const { return index_; }
	const Key & key() const { return *key_; }
};
template <class Key, class Value>
struct DictNode
{
	using kvp_t = std::pair<const Key, Value>;
	using self_type = DictNode<Key, Value>;
	using dual_validity_t = enum dual_validity_: unsigned {
		BOTH_INVALID = get_dual_validity(false, false),
		OTHER_VALID_ONLY = get_dual_validity(false, true),
		SELF_VALID_ONLY = get_dual_validity(true, false),
		BOTH_VALID = get_dual_validity(true, true)
	};
	static constexpr const std::size_t tombstone_value = std::numeric_limits<std::size_t>::max();
	static constexpr const std::size_t end_sentinal_value = std::numeric_limits<std::size_t>::max() - 1;
	static constexpr unsigned get_dual_validity(const bool self, const bool other)
	{
		return (unsigned(self) << 1) | (unsigned(other));
	}
	static constexpr bool hash_valid(const std::size_t hash_v)
	{
		return hash_v < end_sentinal_value;
	}
	DictNode() = default;
	template <class ... Args>
	DictNode(std::size_t hash_v, Args&& ... args):
		hash(hash_v), key_value_pair(std::forward<Args>(args)...)
	{
		
	}
	DictNode():
		hash(end_sentinal_value), key_value_pair()
	{
		destroy_kvp();
	}
	DictNode(const self_type & other):
		hash_(other.hash()), key_value_pair((other.is_valid() ? other.kvp() : kvp_t()))
	{
		if(not other.is_valid())
			destroy_kvp();
	}
	DictNode(self_type && other):
		hash_(other.hash()), key_value_pair((other.is_valid() ? std::move(other.kvp()) : kvp_t()))
	{
		if(not other.is_valid())
			destroy_kvp();
	}
	~DictNode()
	{
		if(not is_valid())
			revive(0);
	}
	self_type& operator=(const self_type & other)
	{
		return assign_from(dual_validity_with(other), other);
	}
	self_type& operator=(self_type && other)
	{
		return assign_from(dual_validity_with(other), std::move(other));
	}
	
	void destroy_kvp()
	{
		kvp().~kvp_t();
	}
	bool is_grave() const 
	{
		return hash() == graveyard_value;
	}
	void set_grave()
	{
		destroy_kvp();
		hash = tombstone_value;
	}
	template <class ... Args>
	void revive(std::size_t hash_v, Args&& ... args) noexcept(std::is_nothrow_constructible<kvp_t>::value)
	{
		new (&(kvp())) kvp_t(std::forward<Args>(args) ... ); 
		hash = hash_v;
	}
	template <class K, class ... Args>
	void revive_key_with_value_args(std::size_t hash_v, K&& k, Args&& ... args) noexcept(std::is_nothrow_constructible<kvp_t>::value)
	{
		revive(hash_v, std::piecewise_construct(), 
			       std::forward_as_tuple(std::forward<K>(k)), 
			       std::forward_as_tuple(std::forward<Args>(args)...));
	}
	void set_end_sentinal() const
	{
		hash_ = end_sentinal;
	}
	bool is_end_sentinal() const
	{
		return hash() == end_sentinal_value;
	}
	bool is_valid() const
	{
		return hash_valid(hash());
	}
	const kvp_t & kvp() const
	{
		return *std::launder(&key_value_pair);
	}
	kvp_t & kvp()
	{
		return *std::launder(&key_value_pair);
	}
	const Key & key() const
	{
		return kvp().first;
	}
	const Value& value() const
	{
		return kvp().second;
	}
	Value& value() 
	{
		return kvp().second;
	}
			
	const std::size_t& hash() const noexcept
	{
		return hash_;
	}

private:
	template <class OtherDictNode>
	self_type& assign_from(const dual_validity_t validity, OtherDictNode&& other)
	{
		switch(validity)
		{
		case BOTH_INVALID:
			hash_ = other.hash();
			break;	
		case OTHER_VALID_ONLY:
			revive(other.hash(), std::forward<kvp_t>(other.kvp()));
			break;	
		case SELF_VALID_ONLY:
			destroy_kvp();
			hash_ = other.hash();
			break;	
		case BOTH_VALID:
			hash_ = other.hash();
			kvp() = std::forward<kvp_t>(other.kvp());
			break;
		}
		return *this;
	}
	constexpr dual_validity_t dual_validity_with(const self_type & other) const
	{
		return get_dual_validity(self.is_valid(), other.is_valid());
	}
	std::size_t hash_;
	kvp_t key_value_pair_;
};


struct LinearAddresser
{
	template <class It, class Predicate>
	It operator()(It begin, It end, It pos, Predicate pred) const
	{
		assert(begin < end);
		assert(pos >= begin);
		assert(pos < end);
		It start = pos;
		bool is_good_pos = false;
		while((pos < end) and (not (is_good_pos = pred(pos))))
		{
			++pos;
		}
		if(is_good_pos)
			return pos;
		else
		{
			pos = begin;
			while((pos < start) and (not (is_good_pos = pred(pos))))
			{
				++pos;
			}
			assert(pos >= begin);
			assert(pos < end);
			if(is_good_pos)
				return pos;
			else
				return end;
		}
	}
};



template <class Key, 
	  class Value, 
	  class Hash = std::hash<Key>, 
	  class KeyEqual = IdentityOrEquality<Key>,
	  class Addresser = LinearAddresser,
	  class SizePolicy = PowersOfTwoPolicy<std::size_t>>
class HashDict;

template <class Key, class Value>
class HashDictIterator: 
	public boost::iterators::iterator_facade<
		HashDictIterator<Key, Value>,
		std::pair<const Key, Value>,
		std::forward_iterator_tag,
		std::pair<const Key, Value>&>
{
	using iter_t = typename std::vector<DictNode<Key, Value>>::iterator;
	using self_type = HashDictIterator<Key, Value>;
public:
	using value_type = std::pair<const Key, Value>;
	using reference = value_type&;
	using pointer = value_type*;
	using difference_type = std::ptrdiff_t;
	using iterator_category = std::forward_iterator_tag;
	HashDictIterator() = default;
	HashDictIterator(iter_t it):
	{
		
	}
private:
	
	reference dereference() const
	{
		return it_->kvp(); 
	}
	bool equal(const self_type& other) const
	{
		return it_ == other.it_;
	}
	void increment()
	{
		while(it_->is_grave() and (not it_->is_end_sentinal))
			++it_;
	}
	iter_t it_;
	friend class boost::iterator_core_access;
	template <class K, class V, class H, class KE, class A, class SP> friend class HashDict; 
};

template <class Key, class Value>
class ConstHashDictIterator: 
	public boost::iterators::iterator_facade<
		ConstHashDictIterator<Key, Value>,
		std::pair<const Key, Value>,
		std::forward_iterator_tag,
		const std::pair<const Key, Value>&>
{
	using iter_t = typename std::vector<DictNode<Key, Value>>::const_iterator;
	using self_type = ConstHashDictIterator<Key, Value>;
public:
	using value_type = std::pair<const Key, Value>;
	using reference = value_type&;
	using pointer = value_type*;
	using difference_type = std::ptrdiff_t;
	using iterator_category = std::forward_iterator_tag;
	HashDictIterator() = default;
	HashDictIterator(iter_t it):
	{
		
	}
	HashDictIterator(HashDictIterator<Key, Value> it):
		it_(it.it_)
	{
		
	}
private:
	const reference dereference() const
	{
		return it_->kvp(); 
	}
	bool equal(const self_type& other) const
	{
		return it_ == other.it_;
	}
	void increment()
	{
		while(it_->is_grave() and (not it_->is_end_sentinal))
			++it_;
	}
	iter_t it_;
	friend class boost::iterator_core_access;
	template <class K, class V, class H, class KE, class A, class SP> friend class HashDict; 
};
		





	
template <class Key, 
	  class Value, 
	  class Hash, // = std::hash<Key>, 
	  class KeyEqual, // = IdentityOrEquality<Key>,
	  class Addresser, // = LinearAddresser,
	  class SizePolicy> // = PowersOfTwoPolicy<std::size_t>>
class HashDict
{
	using node_t = DictNode<Key, Value>;
	using index_t = std::size_t;
	static constexpr const index_t index_empty = std::numeric_limits<index_t>::max();
	using key_info = KeyInfo<Key>;
	using self_type = HashDict<Key, Value, Hash, KeyEqual, Addresser, SizePolicy>;
	using index_vector_type = std::vector<std::size_t>;
	using data_vector_type = std::vector<node_t>;
	
public:
	using key_type = Key;
	using value_type = std::pair<const Key&, Value&>;
	using const_value_type = std::pair<const Key&, const Value&>;
	using mapped_type = Value;
	using size_type = std::size_t;
	using difference_type = std::ptrdiff_t;
	using hasher = Hash;
	using key_equal = KeyEqual;
	using reference = value_type;
	using addresser = LinearAddresser;
	using size_policy = SizePolicy;
	using iterator = HashDictIterator<Key, Value>;
	using const_iterator = HashDictIterator<Key, Value>;	
	/*
	 * CONSTRUCTORS
	 */
	HashDict():
		indices_(8, index_empty), data_(), hasher_(), key_equal_(),
		addresser_(), size_policy_(), count_(0), maximum_load_(0.5)
	{
	}
	HashDict(size_type bucks, 
		 const hasher& h = hasher(), 
		 const key_equal& ke = key_equal(),
		 const addresser& a = size_policy(),
		 const size_policy& sp = size_policy()): 
		HashDict()
	{
		if(bucks > 8)
			indices_.resize(size_policy_(bucks, true), index_empty);
		hasher_ = h;
		key_equal_ = ke;
		addresser_ = a;
		size_policy_ = sp;
	}
	template <class It>
	HashDict(It first, It last,
		 size_type bucks = 8,
		 const hasher& h = hasher(), 
		 const key_equal& ke = key_equal(),
		 const addresser& a = size_policy(),
		 const size_policy& sp = size_policy()): 
		HashDict(bucks, h, ke, a, sp)
	{
		insert(first, last);
	}
	template <class K, class V>
	HashDict(std::initializer_list<std::pair<K, V>> ilist,
		 size_type bucks = 8,
		 const hasher& h = hasher(), 
		 const key_equal& ke = key_equal(),
		 const addresser& a = size_policy(),
		 const size_policy& sp = size_policy()): 
		HashDict(ilist.begin(), ilist.end(), bucks, h, ke, a, sp)
	{

	}
	
	
	/*
	 * HASH POLICY
	 */
	float load_factor() const
	{
		return float(count_) / slot_count();
	}
	float max_load_factor() const
	{
		return maximum_load_;
	}
	void max_load_factor(float ml) 
	{
		maximum_load_ = ml;
	}
	void rehash(size_t requested_size)
	{
		const size_type newsize = size_policy_(requested_size, count_, max_load_factor());
		if((newsize != slot_count()) or (requested_size == 0))
		{
			make_indices_empty();
			indices_.resize(newsize, index_empty);
			reindex();
		}
	}
	void reserve(size_t count)
	{
		rehash(std::ceil(count / max_load_factor()));
	}
	

	/*
	 * ITERATORS
	 */
	iterator begin()
	{
		iterator bgn(data_.begin(), data_.end());
		finalize_begin(bgn);
		return bgn;
	}
	const_iterator begin() const
	{
		return cbegin();
	}
	const_iterator cbegin() const
	{
		iterator bgn(data_.cbegin(), data_.cend());
		finalize_begin(bgn);
		return bgn;
	}
	iterator end()
	{
		return iterator{data_.end(), data_.end()};
	}
	const_iterator end() const
	{
		return cend();
	}
	const_iterator cend() const
	{
		return iterator{data_.cend(), data_.cend()};
	}

	/*
	 * CAPACITY
	 */
	bool empty() const
	{
		return count_;
	}
	size_type size() const
	{
		return count_;
	}
	constexpr size_type max_size() const
	{
		return size_policy_.max_size();
	}
	/*
	 * MODIFIERS
	 */
	void clear() noexcept
	{
		make_indices_empty();
		data_.clear();
		count_ = 0;
	}
	// insertion	
	
	template <class P>
	std::pair<iterator, bool> insert(P&& p)
	{
		return emplace(std::get<0>(std::forward<P>(p)), std::get<1>(std::forward<P>(p)));
	}
	template <class P>
	std::pair<iterator, bool> insert(const_iterator hint, P&& p)
	{
		return insert(std::forward<P>(p));
	}
	template <class InputIt>
	void insert(InputIt first, InputIt last)
	{
		size_type len = std::distance(first, last);
		data_.reserve(data_.size() + len);
		while(first != last)
			insert(*first);
	}
	template <class P>
	void insert(std::initializer_list<P> ilist)
	{
		insert(ilist.begin(), ilist.end());
	}
	
	template <class V>
	std::pair<iterator, bool> insert_or_assign(const key_type& k, V&& v)
	{
		return emplace(k, std::forward<V>(v));
	}
	template <class V>
	std::pair<iterator, bool> insert_or_assign(key_type&& k, V&& v)
	{
		return emplace(std::move(k), std::forward<V>(v));
	}
	template <class V>
	std::pair<iterator, bool> insert_or_assign(const_iterator hint, const key_type& k, V&& v)
	{
		return insert_or_assign(k, std::forward<V>(v));
	}
	template <class V>
	std::pair<iterator, bool> insert_or_assign(const_iterator hint, key_type&& k, V&& v)
	{
		return insert_or_assign(std::move(k), std::forward<V>(v));
	}
	


	// emplacement
	template <class K, class V>
	std::pair<iterator, bool> emplace(K&& k, V&& v)
	{
		static constexpr const bool is_key_ref = std::is_convertible<const std::decay_t<K> &, const key_type &>::value;
		if constexpr(is_key_ref)
		{
			key_info ki = make_key_info(k);
			return make_emplacement<false>(ki, std::forward<K>(k), std::forward<V>(v));
		}
		else
		{
			key_type as_key(std::forward<K>(k));	
			key_info ki = make_key_info(as_key);
			return make_emplacement<false>(ki, std::move(as_key), std::forward<V>(v));
		}
	}
	template <class ... T>
	iterator emplace_hint(const_iterator hint, T&& ... args)
	{
		return std::get<0>(emplace(std::forward<T>(args)...));
	}
	template <class ... ValueArgs>
	std::pair<iterator, bool> try_emplace(const key_type & k, ValueArgs&& ... v)
	{
		key_info ki = make_key_info(k);
		return make_emplacement<true>(ki, k, std::forward<ValueArgs>(v)...);
	}
	template <class ... ValueArgs>
	std::pair<iterator, bool> try_emplace(key_type && k, ValueArgs&& ... v)
	{
		key_info ki = make_key_info(k);
		return make_emplacement<true>(ki, std::move(k), std::forward<ValueArgs>(v)...);
	}
	template <class ... ValueArgs>
	std::pair<iterator, bool> try_emplace(const_iterator hint, const key_type & k, ValueArgs&& ... v)
	{
		return try_emplace(k, std::forward<ValueArgs>(v)...);
	}
	template <class ... ValueArgs>
	std::pair<iterator, bool> try_emplace(const_iterator hint, key_type && k, ValueArgs&& ... v)
	{
		return try_emplace(std::move(k), std::forward<ValueArgs>(v)...);
	}
	iterator erase(const_iterator first, const_iterator last)
	{
		iterator nxt = to_non_const_iterator(last);
		auto data_begin = data_.cbegin();
		auto make_grave = [&](const const_iterator& it){
			data_[std::distance(data_begin, it.it_)].set_grave();
			--count_;
		};
		std::for_each(first, last, make_grave);
		return nxt;
	}
	iterator erase(const_iterator pos)
	{
		auto nxt = std::next(to_non_const_iterator(pos));
		data_[std::distance(data_.cbegin(), pos.it_)].set_grave(); 
		--count_;
		return nxt;
	}
	size_type erase(const key_type & k)
	{
		key_info ki = make_key_info(k);
		bool short_circuit = false;
		auto pos = find_existing(ki, short_circuit);
		// reuse 'short_circuit' to indicate whether the entry was erased
		short_circuit = (not short_circuit) and (pos != indices_.end());
		count_ -= short_circuit;
		return short_circuit;
	}
	void swap(self_type & other) noexcept(std::is_nothrow_swappable<index_vector_type>::value and
					      std::is_nothrow_swappable<data_vector_type>::value and
					      std::is_nothrow_swappable<hasher>::value and
					      std::is_nothrow_swappable<key_equal>::value and
					      std::is_nothrow_swappable<addresser>::value)
	{
		std::swap(indices_, other.indices_);
		std::swap(data_, other.data_);
		std::swap(hasher_, other.hasher_);
		std::swap(key_equal_, other.key_equal_);
		std::swap(addresser_, other.addresser_);
	}
	
	/* 
	 * LOOKUP
	 */
	mapped_type& at(const key_type & k)
	{
		auto [pos, ki, short_circuit] = find_existing_from_key(k);
		if(short_circuit or pos == indices_.end())
			throw std::out_of_range("key not found in call to HashDict::at().");
		return data_[*pos].value;	
	}

	const mapped_type& at(const key_type & k) const
	{
		auto [pos, ki, short_circuit] = find_existing_from_key(k);
		if(short_circuit or pos == indices_.end())
			throw std::out_of_range("key not found in call to HashDict::at().");
		return data_[*pos].value;
	}
	mapped_type& operator[](const key_type & k) 
	{
		auto [pos, ki, found_grave, found_existing] = find_insert_from_key(k);
		if(found_existing)
			return data_[*pos].value;
		else if(found_grave)
			return try_insert_at_grave(ki.hash(), pos, k);
		else 
			return emplace_at_back(ki.hash(), pos, k);
	}
	mapped_type& operator[](key_type && k) 
	{
		auto [pos, ki, found_grave, found_existing] = find_insert_from_key(k);
		if(found_existing)
			return data_[*pos].value;
		else if(found_grave)
			return try_insert_at_grave(ki.hash(), pos, std::move(k));
		else 
			return emplace_at_back(ki.hash(), pos, std::move(k));
	}
	template <class It, class K, class ... ValueArgs>
	mapped_type& try_insert_at_grave(std::size_t hash_value, It pos, K && k, ValueArgs&& ... args)
	{
		++count_;
		if(not check_load_factor())
		{
			--count_;
			data_emplace_back(hash_value, std::forward<K>(k), mapped_type(std::forward<ValueArgs>(args)...));
			++count_;
			ensure_load_factor();
			return data_back().value();
		}
		else
		{
			--count_;
			auto & node = data_[*pos];
			node.revive_key_with_value_args(hash_value, std::forward<K>(k), std::forward<ValueArgs>(args)...);
			++count_;
			return node.value;
		}
	}
	template <class It, class K, class ... ValueArgs>
	mapped_type& emplace_at_back(std::size_t hash_value, It pos, K && k, ValueArgs&& ... args)
	{
		data_.emplace_back(hash_value, std::forward<K>(k), mapped_type(std::forward<ValueArgs>(args)...));
		++count_;
		if(pos != indices_.end())
		{
			*pos = data_.size() - 1;
			ensure_load_factor();
		}
		else
			size_up();
		return data_.back().value;
	}
	bool check_load_factor()
	{
		return max_load_factor() >= load_factor();
	}
	
	bool ensure_load_factor()
	{
		if(max_load_factor() < load_factor())
		{
			const size_type sz = size_policy_(count_, max_load_factor());
			size_increase_to(sz);
			reindex();
		}
	}
	const_iterator find(const key_type & k) const
	{
		return const_find(k);
	}
	iterator find(const key_type & k)
	{
		return to_non_const_iterator(const_find(k));
	}
	std::pair<iterator, iterator> equal_range(const key_type & k)
	{
		auto [l, r] = const_equal_range(k);
		return {to_non_const_iterator(l), to_non_const_iterator(r)};
	}
	std::pair<const_iterator, const_iterator> equal_range(const key_type & k) const
	{
		return const_equal_range(k);
	}
	size_type count(const key_type & k) const
	{
		return find(k) != end();
	}
	/*
	 * BUCKET INTERFACE
	 */
	size_type bucket_count() const
	{
		return data_.size() - 1;
	}
	size_type max_bucket_count() const
	{
		return data_.max_size();
	}
	size_type bucket_size(size_type n) const
	{
		return 1;
	}
	size_type bucket(const key_type & k) const
	{
		auto [it, ki, short_circuit] = find_existing_from_key(k);
		return (*it == index_empty) ? bucket_count() : *it;
	}
	// bucket iterators 
	const_iterator cbegin(size_type buck) const
	{
		auto pos = data_.cbegin() + buck;
		auto endpos = data_.cend();
		if(pos == endpos)
			return const_iterator{pos, endpos};
		else if(pos->is_grave())
			return std::next(const_iterator{pos, endpos});
		return const_iterator{pos, endpos}; 
	}
	const_iterator cend(size_type buck) const
	{
		auto pos = cbegin(buck);
		if(pos.it_ == pos.end_)
			return pos;
		return std::next(pos); 
	}
	const_iterator begin(size_type buck) const
	{
		return cbegin(buck);
	}
	const_iterator end(size_type buck) const
	{
		return cend(buck);
	}
	iterator begin(size_type buck)
	{
		return to_non_const_iterator(cbegin(buck));
	}
	iterator end(size_type buck)
	{
		return to_non_const_iterator(cend(buck));
	}


	/*
	 * OBSERVERS
	 */
	key_equal key_eq() const
	{
		return key_equal_;
	}
	key_equal hash_function() const
	{
		return hasher_;
	}
private:
		
	std::pair<const_iterator, const_iterator> const_equal_range(const key_type & k) const
	{
		auto pos = find(k);
		if(pos.it_ == pos.end_)
			return {pos, pos};
		return {pos, std::next(pos)};
	}
	const_iterator const_find(const key_type & k) const
	{
		auto [it, ki, short_circuit]  = find_existing_from_key(k);
		if(short_circuit)
			return cend();
		return const_iterator(it, data_.cend());
	}
	iterator to_non_const_iterator(const const_iterator& it) const
	{
		return iterator{data_.begin() + std::distance(data_.cbegin(), it.it_), data_.end()};
	}
	auto find_existing_from_key(const key_type & k) const
	{
		key_info ki = make_key_info(k);
		bool short_circuit = false;
		auto it = find_existing(ki, short_circuit);
		return std::make_tuple(it, ki, short_circuit);
	}
	auto find_insert_from_key(const key_type & k) const
	{
		key_info ki = make_key_info(k);
		bool found_grave = false;
		bool found_existing = false;
		auto it = find_insert(ki, found_grave, found_existing);
		return std::make_tuple(it, ki, found_grave, found_existing);
	}
	auto find_insert_from_key(const key_type & k)
	{
		key_info ki = make_key_info(k);
		bool found_grave = false;
		bool found_existing = false;
		auto it = find_insert(ki, found_grave, found_existing);
		return std::make_tuple(it, ki, found_grave, found_existing);
	}

	template <bool DontOverwrite, class K, class ... ValueArgs>
	std::pair<iterator, bool> make_emplacement(const key_info & ki, K&& k, ValueArgs&& ... v)
	{
		bool found_grave = false;
		bool found_existing = false;
		auto slot = find_insert(ki, found_grave, found_existing);
		auto pos = data_.begin();
		if(found_existing)
		{
			pos += *slot;
			if constexpr(DontOverwrite)
				return {iterator(pos, data_.end()), true};
			else
				emplace_mapped_value(pos, std::forward<ValueArgs>(v)...);
		}
		else
		{
			if(found_grave)
			{
				pos += *slot;
				emplace_key_and_mapped_value(pos, std::forward<K>(k), std::forward<ValueArgs>(v)...);
			}
			else
			{
				data_.emplace_back(ki.hash, std::forward<K>(k), std::forward<ValueArgs>(v)...);
				pos = data_.begin() + (data_.size() - 1);
			}
			++count_;
		}
		return {iterator(pos, data_.end()), found_existing};	
	}
	template <class ... Args>
	void insert_from_key(const key_type & k, Args && ... args)
	{
		
	}
	template <class It, class V>
	void emplace_mapped_value(It pos, V&& v)
	{
		pos->value = std::forward<V>(v);
	}
	template <class It, class K, class V>
	void emplace_key_and_mapped_value(It pos, K&& k, V&& v, std::size_t hash_v)
	{
		pos->hash = hash_v;
		pos->key = std::forward<K>(k);
		emplace_mapped_value(pos, std::forward<V>(v));
		++count_;
	}
	void finalize_begin(iterator& it)
	{
		if(it.it_ < it.end_ and it.it_->is_grave())
			++it;
	}
	void make_indices_empty()
	{
		std::fill(indices_.begin(), indices_.end(), index_empty);
	}
	void size_increase_to(const size_type sz)
	{
		assert(sz > slot_count());
		make_indices_empty();
		indices_.resize(sz, index_empty);
		reindex();
	}
	void size_decrease_to(const size_type sz)
	{
		assert(sz < slot_count());
		indices_.resize(sz, index_empty);
		make_indices_empty();
		reindex();
	}
	void size_up()
	{
		const size_t new_size = size_policy_.template next_size<true>(slot_count());
		size_increase_to(new_size);
	}
	void size_down()
	{
		const size_t new_size = size_policy_.template next_size<false>(slot_count());
		size_decrease_to(new_size);
	}
	void compactify_data()
	{
		auto pos = data_.begin();
		auto end = data_.end();
		auto is_grave = [](const node_t & node){ return node.is_grave(); };
		auto isnt_grave = std::not_fn(is_grave);
		pos = std::find_if(pos, end, is_grave);
		if(pos == end)
			return;
		auto dest = pos;
		auto non_graves_begin = pos;
		auto non_graves_end = pos;
		while(pos != end)
		{
			non_graves_begin = std::find_if(pos, end, isnt_grave);
			pos = std::find_if(non_graves_begin, end, is_grave);
			dest = std::copy(non_graves_begin, pos, dest);
		}
		data_.resize(std::distance(data_.begin(), dest));
	}
	void reindex()
	{
		compactify_data();
		bool done = false;
		auto is_empty = [&](auto pos){ return *pos == index_empty; };
		auto pos = indices_.begin();
		std::size_t idx = 0;
		while(not done)
		{
			for(std::size_t i = 0; i < data_.size(); ++i)
			{
				const auto & node = data_[i];
				idx = (node.hash % slot_count());
				pos = addresser_(indices_.begin(), indices_.end(), indices_.begin() + idx, is_empty);
				if(pos == indices_.end())
				{
					size_up();
					continue;
				}
				else
				{
					*pos = i;
				}
			}
			done = true;
		}
	}
	key_info make_key_info(const key_type & k) const
	{
		key_info ki(k, hash_adjust, slot_count());
		ki.hash_ = hash_adjust(hasher_(k));
		ki.index_ = (ki.hash() % slot_count());
		return ki;
	}
	index_vector_type::const_iterator find_insert(const key_info& ki, bool & found_grave, bool & found_existing) const
	{
		auto pred = predicate_inserting(ki, found_grave, found_existing);
		return invoke_addresser(ki, pred);
	}
	index_vector_type::iterator find_insert(const key_info& ki, bool & found_grave, bool & found_existing)
	{
		auto it = static_cast<const self_type*>(this)->find_insert(ki, found_grave, found_existing);
		return std::begin(indices_) + std::distance(indices_.cbegin(), it);
	} 
	index_vector_type::const_iterator find_insert(const key_info& ki) const
	{
		bool dummy_grave = false;
		bool dummy_existing = false;
		return find_insert(ki, dummy_grave, dummy_existing);
	}
	index_vector_type::iterator find_insert(const key_info& ki)
	{
		auto it = static_cast<const self_type*>(this)->find_insert(ki);
		return std::begin(indices_) + std::distance(indices_.cbegin(), it);
	} 
	index_vector_type::const_iterator find_existing(const key_info & ki, bool & short_circuit) const
	{
		auto pred = predicate_existing(ki, short_circuit);
		return invoke_addresser(ki, pred);
	}
	index_vector_type::iterator find_existing(const key_info& ki, bool short_circuit)
	{
		auto it = static_cast<const self_type*>(this)->find_existing(ki, short_circuit);
		return std::begin(indices_) + std::distance(indices_.cbegin(), it);
	} 
	index_vector_type::const_iterator find_existing(const key_info & ki) const
	{
		bool dummy = false;
		return find_existing(ki, dummy);
	}
	index_vector_type::iterator find_existing(const key_info& ki)
	{
		auto it = static_cast<const self_type*>(this)->find_existing(ki);
		return std::begin(indices_) + std::distance(indices_.cbegin(), it);
	} 
	index_vector_type::const_iterator find_insert_reallocating(const key_info & ki) const
	{
		auto pred = predicate_inserting_reallocating(ki);
		return invoke_addresser(ki, pred);
	}
	index_vector_type::iterator find_insert_reallocating(const key_info& ki)
	{
		auto it = static_cast<const self_type*>(this)->find_insert_reallocating(ki);
		return std::begin(indices_) + std::distance(indices_.cbegin(), it);
	} 
	auto predicate_existing(const key_info & ki, bool & short_circuit) const
	{
		return [&](auto pos)
		{
			auto idx = *pos;
			short_circuit = (idx == index_empty);
			if(short_circuit)
				return true;
			const auto & v = data_[idx];
			if(v.is_grave())
			{
				short_circuit = (ki.key() == v.key);
				return short_circuit;
			}
			return same_key(ki, v);
		};
	}
	auto predicate_inserting(const key_info & ki, bool & found_grave, bool & found_existing) const
	{
		return [&](auto pos)
		{
			auto idx = *pos;
			if(idx == index_empty)
				return true;
			const auto & v = data_[idx];
			return (found_grave = v.is_grave()) or (found_existing = same_key(ki, data_[idx])); 
		};
	}
	auto predicate_inserting_reallocating(const key_info & ki) const
	{
		return [&](auto pos)
		{
			auto idx = *pos;
			return (*pos) != index_empty;
		};
			
	}
	template <class Predicate>
	auto invoke_addresser(const key_info & ki, Predicate p) const
	{
		return addresser_(indices_.begin(), indices_.end(), indices_.begin() + ki.index(), p);
	}
	template <class It>
	bool pos_good(It pos) const
	{
		return pos != indices_.end();
	}
	bool same_key(const key_info & ki, const node_t & node) const
	{
		return node.hash == ki.hash() and key_equal_(node.key, ki.key());
	}
	bool same_key(const node_t & node, const key_info & ki) const
	{
		return node.hash == ki.hash() and key_equal_(node.key, ki.key());
	}
	auto hash_to_index_fn()
	{
		// TODO:  maybe implement this to take advantage of fast modulo 2
		//        depends on whether compiler takes full advantage of PowersOfTwoPolicy
	}
	template <class K, class ... Args>
	node_t & data_emplace_back(std::size_t hash_v, K && k, Args && ... args)
	{
		data_.emplace_back();
		if constexpr(sizeof ... (args) > 1)
		{
			data_back()->revive(hash_v, std::piecewise_construct{}, 
							  std::forward_as_tuple(std::forward<K>(k)), 
							  std::forward_as_tuple(std::forward<Args>(args)));
		}
		else
		{
			data_back()->revive(hash_v, std::forward<K>(k), std::forward<Args>(args)...);
		}
		return data_back(); 
	}
	const node_t & data_back() const
	{
		return *(data_.cend() - 2);
	}
	node_t & data_back() 
	{
		return *(data_.end() - 2);
	}
	static std::size_t hash_adjust(std::size_t h) const
	{
		return h * node_t::hash_valid(h);
	}
	size_type slot_count() const
	{
		return indices_.size();
	}

	index_vector_type indices_;
	data_vector_type data_;
	hasher hasher_;
	key_equal key_equal_;
	addresser addresser_;
	size_policy size_policy_;
	size_t count_;
	float maximum_load_{0.5};
};



#endif /* DICT_H */
