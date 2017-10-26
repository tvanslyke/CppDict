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
#include <new>
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
		hash_(hash_v), index_((hash_v % container_length)), key_(&k)
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
	static constexpr unsigned get_dual_validity(const bool self, const bool other)
	{
		return (unsigned(self) << 1) | (unsigned(other));
	}
	using dual_validity_t = enum dual_validity_: unsigned {
		BOTH_INVALID = get_dual_validity(false, false),
		OTHER_VALID_ONLY = get_dual_validity(false, true),
		SELF_VALID_ONLY = get_dual_validity(true, false),
		BOTH_VALID = get_dual_validity(true, true)
	};
	static constexpr const std::size_t tombstone_value = std::numeric_limits<std::size_t>::max();
	static constexpr const std::size_t end_sentinal_value = std::numeric_limits<std::size_t>::max() - 1;
	static constexpr bool hash_valid(const std::size_t hash_v)
	{
		return hash_v < end_sentinal_value;
	}
	

	DictNode():
		hash_(end_sentinal_value), key_value_pair_()
	{
		destroy_kvp();
	}
	template <class ... Args>
	DictNode(std::size_t hash_v, Args&& ... args):
		hash_(hash_v), key_value_pair_(std::forward<Args>(args)...)
	{
		
	}
	DictNode(const self_type & other):
		hash_(other.hash()), key_value_pair_((other.is_valid() ? other.kvp() : kvp_t()))
	{
		if(not other.is_valid())
			destroy_kvp();
	}
	DictNode(self_type && other):
		hash_(other.hash()), key_value_pair_((other.is_valid() ? std::move(other.kvp()) : kvp_t()))
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
		return hash() == tombstone_value;
	}
	void set_grave()
	{
		destroy_kvp();
		hash_ = tombstone_value;
	}
	template <class ... Args>
	void revive(std::size_t hash_v, Args&& ... args) noexcept(std::is_nothrow_constructible<kvp_t>::value)
	{
		new (&(kvp())) kvp_t(std::forward<Args>(args) ... ); 
		hash_ = hash_v;
	}
	template <class K, class ... Args>
	void revive_key_with_value_args(std::size_t hash_v, K&& k, Args&& ... args) noexcept(std::is_nothrow_constructible<kvp_t>::value)
	{
		if constexpr(sizeof...(Args) == 0)
		{
			revive(hash_v, std::forward<K>(k), Value());
		}
		else
		{
			revive(hash_v, std::piecewise_construct_t(), 
				       std::forward_as_tuple(std::forward<K>(k)), 
				       std::forward_as_tuple(std::forward<Args>(args)...));
		}
	}
	template <class Hasher, class ... Args>
	void revive_hash_inplace(Hasher hasher, Args&& ... args)
	{
		revive(tombstone_value, std::forward<Args>(args)...);
		hash_ = hasher(key());
	}
	void set_end_sentinal() const
	{
		assert(not is_valid());
		hash_ = end_sentinal_value;
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
		return *std::launder(&key_value_pair_);
	}
	kvp_t & kvp()
	{
		return *std::launder(&key_value_pair_);
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
			revive(other.hash(), std::forward<OtherDictNode>(other).kvp());
			break;	
		case SELF_VALID_ONLY:
			destroy_kvp();
			hash_ = other.hash();
			break;	
		case BOTH_VALID:
			set_grave();
			revive(other.hash(), std::forward<OtherDictNode>(other).kvp());
			break;
		}
		return *this;
	}
	constexpr dual_validity_t dual_validity_with(const self_type & other) const
	{
		return static_cast<dual_validity_t>(get_dual_validity(is_valid(), other.is_valid()));
	}
	std::size_t hash_;
	kvp_t key_value_pair_;
};


struct LinearAddresser
{
	template <class It, class Predicate>
	It operator()(It begin, It end, It pos, Predicate pred) const
	{
		It start = pos;
		bool is_good_pos = false;
		while((pos < end) and (not (is_good_pos = pred(pos))))
			++pos;
		if(is_good_pos)
			return pos;
		else
		{
			pos = begin;
			while((pos < start) and (not (is_good_pos = pred(pos))))
				++pos;
			if(is_good_pos)
				return pos;
			else
				return end;
		}
	}
	
	template <class It, class Predicate>
	It operator()(It begin, It end, It pos, It start, Predicate pred) const
	{
		bool is_good_pos = false;
		while((pos < end) and (pos < start) and (not (is_good_pos = pred(pos))))
			++pos;
		if(is_good_pos)
			return pos;
		else if(pos != start)
		{
			pos = begin;
			while((pos < start) and (not (is_good_pos = pred(pos))))
				++pos;
			if(is_good_pos)
				return pos;
			else
				return end;
		}
		else
			return end;
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
		it_(it)
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
		++it_;
		incr_until_valid_or_end();
	}
	void incr_until_valid_or_end()
	{
		while(it_->is_grave() and (not it_->is_end_sentinal()))
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
	ConstHashDictIterator() = default;
	ConstHashDictIterator(iter_t it):
		it_(it)
	{
		
	}
	ConstHashDictIterator(HashDictIterator<Key, Value> it):
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
		++it_;
		incr_until_valid_or_end();
	}
	void incr_until_valid_or_end()
	{
		while(it_->is_grave() and (not it_->is_end_sentinal()))
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
	static constexpr const std::size_t index_empty = std::numeric_limits<std::size_t>::max();
	using node_t = DictNode<Key, Value>;
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
	using const_iterator = ConstHashDictIterator<Key, Value>;	
	/*
	 * CONSTRUCTORS
	 */
	HashDict():
		indices_(8, index_empty), data_(1), hasher_(), key_equal_(),
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
		return to_non_const_iterator(cbegin());
	}
	const_iterator begin() const
	{
		return cbegin();
	}
	const_iterator cbegin() const
	{
		const_iterator bgn(data_.cbegin());
		bgn.incr_until_valid_or_end();
		return bgn;
	}
	iterator end()
	{
		return to_non_const_iterator(cend());
	}
	const_iterator end() const
	{
		return cend();
	}
	const_iterator cend() const
	{
		return const_iterator(data_.cend() - 1);
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
		data_.resize(1);
		data_.back().set_end_sentinal();
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
		// TODO: use TMP to see if we can check which keys already exist before inserting.
		//       of course measure to see that it's actually faster!
		size_type len = std::distance(first, last);
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
	template <class ... Args>
	std::pair<iterator, bool> emplace(Args&& ... args)
	{
		auto pos = emplace_maybe(std::forward<Args>(args)...);
		auto ki = make_key_info(pos->key(), pos->hash());
		bool found_existing = false;
		auto idx_pos = find_insert_ignore_graves(ki, found_existing);
		if(found_existing)
		{
			undo_emplace_maybe();
			return {iterator(data_.begin() + *idx_pos), true};
		}
		else if(idx_pos != indices_.end())
		{
			*idx_pos = std::distance(data_.begin(), pos);
			++count_;
			ensure_load_factor_no_invalidate_iterators();
			return {iterator(pos), false};
		}
		else
		{
			size_up();
			reindex_no_invalidate_iterators();
			return {iterator(pos), false};
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
		auto pos = first.it_;
		while(first != last)
		{
			++first;
			pos->set_grave();
			--count_;
			pos = first.it_;
		}
		return iterator(pos);
	}
	iterator erase(const_iterator pos)
	{
		// TODO: optimize
		if(pos != cend())
		{
			--count_;
			return erase(pos, std::next(pos));
		}
	}
	size_type erase(const key_type & k)
	{
		auto [pos, ki, short_circuit] = find_existing_from_key(k);
		// reuse 'short_circuit' to indicate whether the entry was erased
		short_circuit = (not short_circuit) and (pos != indices_.end());
		if(short_circuit)
			erase(data_.cbegin() + *pos);
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
		key_info ki = make_key_info(k);
		return make_emplacement<true>(ki, k).first->second;
	}
	mapped_type& operator[](key_type && k) 
	{
		key_info ki = make_key_info(k);
		return make_emplacement<true>(ki, std::move(k)).first->second;
	}
	
	template <class ... Args>
	typename data_vector_type::iterator emplace_maybe(Args&& ... args)
	{
		data_.emplace_back();
		auto adjusted_hasher = [&](const key_type& k){return hash_adjust(hasher_(k));};
		data_back().revive_hash_inplace(adjusted_hasher, std::forward<Args>(args)...);
		return data_last();
	}
	void undo_emplace_maybe()
	{
		data_last().set_end_sentinal();
		data_.pop_back();
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
			return false;
		}
		return true;
	}
	bool ensure_load_factor_no_invalidate_iterators()
	{
		if(max_load_factor() < load_factor())
		{
			const size_type sz = size_policy_(count_, max_load_factor());
			size_increase_to(sz);
			reindex_no_invalidate_iterators();
			return false;
		}
		return true;
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
	iterator to_non_const_iterator(const const_iterator& it) 
	{
		return iterator(data_.begin() + std::distance(data_.cbegin(), it.it_));
	}
	typename index_vector_type::iterator to_non_const_iterator(typename index_vector_type::const_iterator it)
	{
		return indices_.begin() + std::distance(indices_.cbegin(), it);
	}
	typename data_vector_type::iterator to_non_const_iterator(typename data_vector_type::const_iterator it)
	{
		return data_.begin() + std::distance(data_.cbegin(), it);
	}
	auto find_existing_from_key(const key_type & k) const
	{
		key_info ki = make_key_info(k);
		bool short_circuit = false;
		auto it = find_existing(ki, short_circuit);
		return std::make_tuple(it, ki, short_circuit);
	}
	auto find_existing_from_key(const key_type & k) 
	{
		auto [it, ki, sc] = find_existing_from_key(k);
		return std::make_tuple(to_non_const_iterator(it), ki, sc);
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
		auto [it, ki, fg, fe] = static_cast<const self_type*>(this)->find_insert_from_key(k);
		return std::make_tuple(to_non_const_iterator(it), ki, fg, fe);
	}
	auto find_insert_ignore_graves_from_key(const key_type & k) const
	{
		key_info ki = make_key_info(k);
		bool found_existing = false;
		auto it = find_insert_ignore_graves(ki, found_existing);
		return std::make_tuple(it, ki, found_existing);
	}
	auto find_insert_ignore_graves_from_key(const key_type & k) 
	{
		auto [it, ki, fe] = static_cast<const self_type*>(this)->find_insert_ignore_graves_from_key(k);
		return std::make_tuple(to_non_const_iterator(it), ki, fe);
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
				return {iterator(pos), false};
			else
				emplace_existing(pos, std::forward<ValueArgs>(v)...);
		}
		else
		{
			if(found_grave)
			{
				auto terminal_pos = check_for_existing_from(ki, slot);
				if(terminal_pos != indices_.end())
				{
					// TODO:  all this machinery is for debugging.  can just return {iterator(terminal_pos), false}
					auto & node = data_[*terminal_pos];
					assert(not key_equal_(node.key(), ki.key()));
					return {iterator(data_.begin() + *terminal_pos), false};
				}
				else
				{
					pos += *slot;
					emplace_grave(ki.hash(), pos, std::forward<K>(k), std::forward<ValueArgs>(v)...);
					ensure_load_factor_no_invalidate_iterators();
				}
			}
			else
			{
				// careful,  may invalidate 'pos'!  don't do 'pos += (data_.size() - 2);'!
				data_emplace_back(ki.hash(), std::forward<K>(k), std::forward<ValueArgs>(v)...);
				// also may invalidate 'pos'!
				ensure_load_factor();
				pos = data_.begin() + (data_.size() - 2);
			}
		}
		return {iterator(pos), found_existing};	
	}
	
	template <class It, class ... Args>
	void emplace_existing(It pos, Args&& ... args)
	{
		pos->value = mapped_type(std::forward<Args>(args)...);
	}
	template <class It, class K, class ... V>
	void emplace_grave(std::size_t hash_v, It pos, K&& k, V&& ... v)
	{
		pos->revive_key_with_value_args(hash_v, std::forward<K>(k), std::forward<V>(v) ...);
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
		while(pos != end)
		{
			non_graves_begin = std::find_if(pos, end, isnt_grave);
			pos = std::find_if(non_graves_begin, end, is_grave);
			dest = std::copy(non_graves_begin, pos, dest);
		}
		data_.resize(std::distance(data_.begin(), dest));
	}
	void reindex_no_invalidate_iterators()
	{
		bool done = false;
		auto is_empty = [&](auto pos){ return *pos == index_empty; };
		auto pos = indices_.begin();
		std::size_t idx = 0;
		while(not done)
		{
			// TODO: we know 'count_', so we can cut off the operation early if
			//       keep track of how many non-graves we've found so far
			// TODO: DRY!  this is nearly a carbon-copy of 'reindex()'!
			for(std::size_t i = 0; i < data_.size() - 1; ++i)
			{
				const auto & node = data_[i];
				if(node.is_grave())
					continue;
				assert(not node.is_end_sentinal());
				idx = (node.hash() % slot_count());
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
		assert(data_.back().is_end_sentinal());
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
				idx = (node.hash() % slot_count());
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
		return key_info(k, hash_adjust(hasher_(k)), slot_count());
	}
	key_info make_key_info(const key_type & k, std::size_t hash_v) const
	{
		return key_info(k, hash_v, slot_count());
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
	
	index_vector_type::const_iterator check_for_existing_from(const key_info & ki, index_vector_type::const_iterator continue_from) const
	{
		bool short_circuit = false;
		auto pred = predicate_existing(ki, short_circuit);
		auto it = continue_addresser(ki, continue_from, pred);
		if(short_circuit)
			return indices_.end();
		else 
			return it;
	}
	index_vector_type::iterator check_for_existing_from(const key_info & ki, index_vector_type::const_iterator continue_from)
	{
		auto it = static_cast<const self_type*>(this)->check_for_existing_from(ki, continue_from);
		return to_non_const_iterator(it);
	}	
	index_vector_type::const_iterator find_insert_ignore_graves(const key_info& ki, bool & found_existing) const
	{
		auto pred = predicate_inserting_ignore_graves(ki, found_existing);
		return invoke_addresser(ki, pred);
	}
	index_vector_type::iterator find_insert_ignore_graves(const key_info& ki, bool & found_existing)
	{
		auto it = static_cast<const self_type*>(this)->find_insert_ignore_graves(ki, found_existing);
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
	auto predicate_inserting_ignore_graves(const key_info & ki, bool & found_existing) const
	{
		return [&](auto pos)
		{
			auto idx = *pos;
			if(idx == index_empty)
				return true;
			const auto & v = data_[idx];
			return (found_existing = same_key(ki, data_[idx])); 
		};
	}
	template <class Predicate>
	auto invoke_addresser(const key_info & ki, Predicate p) const
	{
		return addresser_(indices_.begin(), indices_.end(), indices_.begin() + ki.index(), p);
	}
	template <class It, class Predicate>
	auto continue_addresser(const key_info& ki, It continue_from, Predicate p) const
	{
		return addresser_(indices_.begin(), indices_.end(), continue_from, indices_.begin() + ki.index(), p);
	}
	template <class It>
	bool pos_good(It pos) const
	{
		return pos != indices_.end();
	}
	bool same_key(const key_info & ki, const node_t & node) const
	{
		return (node.hash() == ki.hash()) and key_equal_(node.key(), ki.key());
	}
	bool same_key(const node_t & node, const key_info & ki) const
	{
		return (node.hash() == ki.hash()) and key_equal_(node.key(), ki.key());
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
		data_back().revive_key_with_value_args(hash_v, std::forward<K>(k), std::forward<Args>(args) ...);
		++count_;
		return data_back(); 
	}
	const node_t & data_back() const
	{
		return *data_last();
	}
	node_t & data_back() 
	{
		return *data_last();
	}
	auto data_last() const
	{
		return data_.cend() - 2;
	}
	auto data_last()
	{
		return data_.end() - 2;
	}
	static std::size_t hash_adjust(std::size_t h) 
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
