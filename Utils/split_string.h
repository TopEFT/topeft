#ifndef SPLITSTRING_H_
#define SPLITSTRING_H_

#include <string>
//#include <boost/algorithm/string.hpp>

template <class Container>
void split_string(const std::string& str, Container& cont, const std::string& delims = " ") {
  int head = 0;
  int tail = 0;
  while(tail < (int)str.length()) {
    tail = str.find(delims, tail);
    if(tail < 0) tail = str.length();
    cont.push_back(string(str.substr(head, tail-head)));
    head = ++tail;
  }
}
/*
// See http://www.martinbroadhurst.com/how-to-split-a-string-in-c.html
void split_string(const std::string& str, Container& cont, const std::string& delims = " ") {
    boost::split(cont,str,boost::is_any_of(delims));
}
*/

#endif
/* SPLITSTRING */
