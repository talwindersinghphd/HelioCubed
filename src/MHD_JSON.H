#pragma once

// Small, strict JSON reader for the versioned dGCS interchange files. No
// runtime Python, mesh/field-line interpolation, or new library is required.
#include <cctype>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace CMEJson {
struct Value {
    enum Kind { Null, Boolean, Number, String, Array, Object } kind = Null;
    double number = 0;
    bool boolean = false;
    std::string string;
    std::vector<Value> array;
    std::map<std::string, Value> object;
    const Value& at(const std::string& key) const {
        auto it = object.find(key);
        if (kind != Object || it == object.end())
            throw std::invalid_argument("Missing JSON field: " + key);
        return it->second;
    }
    bool has(const std::string& key) const { return kind == Object && object.count(key); }
    double real() const {
        if (kind != Number) throw std::invalid_argument("Expected a JSON number");
        return number;
    }
    int integer() const {
        const double v = real();
        if (v != std::floor(v) || std::abs(v) > 1000000000.)
            throw std::invalid_argument("Expected a small JSON integer");
        return static_cast<int>(v);
    }
    const std::string& text() const {
        if (kind != String) throw std::invalid_argument("Expected a JSON string");
        return string;
    }
    const std::vector<Value>& items() const {
        if (kind != Array) throw std::invalid_argument("Expected a JSON array");
        return array;
    }
    bool truth() const {
        if (kind != Boolean) throw std::invalid_argument("Expected a JSON boolean");
        return boolean;
    }
};

class Reader {
    const std::string& input;
    size_t pos = 0;
    [[noreturn]] void fail() const {
        throw std::invalid_argument("Invalid JSON near byte " + std::to_string(pos));
    }
    void space() { while (pos < input.size() && (input[pos]==' ' || input[pos]=='\n' || input[pos]=='\r' || input[pos]=='\t')) ++pos; }
    bool take(char c) { space(); if (pos < input.size() && input[pos] == c) { ++pos; return true; } return false; }
    unsigned hex() {
        unsigned value = 0;
        for (int i=0; i<4; ++i) {
            if (pos == input.size()) fail();
            char c = input[pos++];
            int n = c>='0' && c<='9' ? c-'0' : c>='a' && c<='f' ? c-'a'+10 : c>='A' && c<='F' ? c-'A'+10 : -1;
            if (n < 0) fail();
            value = value*16+n;
        }
        return value;
    }
    std::string text() {
        if (!take('"')) fail();
        std::string result;
        while (pos < input.size()) {
            unsigned char c = input[pos++];
            if (c == '"') return result;
            if (c < 32) fail();
            if (c != '\\') { result += char(c); continue; }
            if (pos == input.size()) fail();
            c = input[pos++];
            switch (c) {
                case '"': case '\\': case '/': result += char(c); break;
                case 'b': result += '\b'; break; case 'f': result += '\f'; break;
                case 'n': result += '\n'; break; case 'r': result += '\r'; break; case 't': result += '\t'; break;
                case 'u': {
                    unsigned code = hex();
                    if (code >= 0xd800 && code <= 0xdbff) {
                        if (pos+2 > input.size() || input[pos++]!='\\' || input[pos++]!='u') fail();
                        unsigned low = hex();
                        if (low < 0xdc00 || low > 0xdfff) fail();
                        code = 0x10000+((code-0xd800)<<10)+(low-0xdc00);
                    } else if (code >= 0xdc00 && code <= 0xdfff) fail();
                    if (code < 0x80) result += char(code);
                    else if (code < 0x800) { result += char(0xc0|(code>>6)); result += char(0x80|(code&63)); }
                    else if (code < 0x10000) { result += char(0xe0|(code>>12)); result += char(0x80|((code>>6)&63)); result += char(0x80|(code&63)); }
                    else { result += char(0xf0|(code>>18)); result += char(0x80|((code>>12)&63)); result += char(0x80|((code>>6)&63)); result += char(0x80|(code&63)); }
                    break;
                }
                default: fail();
            }
        }
        fail();
    }
    Value value(int depth) {
        if (depth > 128) fail();
        space(); if (pos == input.size()) fail();
        Value v;
        if (input[pos]=='{') {
            ++pos; v.kind=Value::Object;
            if (take('}')) return v;
            do {
                auto key=text(); if (!take(':')) fail();
                if (!v.object.emplace(key,value(depth+1)).second) fail();
                if (take('}')) return v;
            } while (take(','));
            fail();
        }
        if (input[pos]=='[') {
            ++pos; v.kind=Value::Array;
            if (take(']')) return v;
            do { v.array.push_back(value(depth+1)); if (take(']')) return v; } while (take(','));
            fail();
        }
        if (input[pos]=='"') { v.kind=Value::String; v.string=text(); return v; }
        for (const std::string token : {"true","false","null"}) {
            if (input.compare(pos,token.size(),token)==0) {
                pos += token.size(); v.kind=token=="null" ? Value::Null : Value::Boolean;
                v.boolean=token=="true"; return v;
            }
        }
        size_t start=pos;
        if (input[pos]=='-') ++pos;
        if (pos==input.size()) fail();
        if (input[pos]=='0') ++pos;
        else { if (input[pos]<'1' || input[pos]>'9') fail(); while (pos<input.size() && std::isdigit(static_cast<unsigned char>(input[pos]))) ++pos; }
        auto digits=[&]() { size_t begin=pos; while(pos<input.size() && std::isdigit(static_cast<unsigned char>(input[pos]))) ++pos; if(pos==begin) fail(); };
        if (pos<input.size() && input[pos]=='.') { ++pos; digits(); }
        if (pos<input.size() && (input[pos]=='e' || input[pos]=='E')) { ++pos; if(pos<input.size() && (input[pos]=='+' || input[pos]=='-')) ++pos; digits(); }
        try { v.number=std::stod(input.substr(start,pos-start)); } catch(...) { fail(); }
        if (!std::isfinite(v.number)) fail();
        v.kind=Value::Number; return v;
    }
  public:
    explicit Reader(const std::string& text) : input(text) {}
    Value parse() { Value result=value(0); space(); if(pos!=input.size()) fail(); return result; }
};

inline Value read(const std::string& path) {
    std::ifstream input(path,std::ios::binary|std::ios::ate);
    if (!input) throw std::invalid_argument("Cannot open CME JSON file: "+path);
    if (input.tellg() > 64*1024*1024) throw std::invalid_argument("CME JSON file exceeds 64 MiB: "+path);
    input.seekg(0); std::ostringstream contents; contents << input.rdbuf();
    const std::string text=contents.str();
    try { return Reader(text).parse(); }
    catch(const std::exception& error) { throw std::invalid_argument(path+": "+error.what()); }
}
} // namespace CMEJson
