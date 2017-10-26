
BOOST_AUTO_TEST_CASE( test_types )
{
	std::cerr << "test_types" << std::endl;
        BOOST_TEST((std::is_same<C::key_type, dict_key_t>::value));
        BOOST_TEST((std::is_same<C::mapped_type, dict_mapped_t>::value));
        BOOST_TEST((std::is_same<C::hasher, std::hash<C::key_type> >::value));
        BOOST_TEST((std::is_same<C::value_type, std::pair<const C::key_type, C::mapped_type> >::value));
        BOOST_TEST((std::is_same<C::reference, C::value_type&>::value));
        BOOST_TEST((std::is_same<C::const_reference, const C::value_type&>::value));
}
