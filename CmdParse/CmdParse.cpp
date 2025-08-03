#include "CmdParse.h"

CmdParse::CmdParse()
{

}

void CmdParse::add(const std::string shortForm, bool hasArg)
{
	this->add(shortForm, "", hasArg);
}

void CmdParse::add(const std::string shortForm, const std::string longForm, bool hasArg)
{
	ArgType arg;
	arg.shortForm = shortForm;
	arg.longForm = longForm;
	arg.hasArg = hasArg;

	_argType.push_back(arg);
}

bool CmdParse::get(const std::string opt, ArgType &t)
{
	for(unsigned int i = 0; i < _argType.size(); i++) {
		if(_argType[i].shortForm == opt || _argType[i].longForm == opt) {
			t = _argType[i];
			return true;
		}
	}

	return false;
}

void CmdParse::parse(int argc, char **argv)
{
        for(int i = 1; i < argc; i++) {
                std::string arg(argv[i]);

                std::string opt = arg;
                std::string optValue;

                // Allow options in the form --option=value
                size_t pos = arg.find('=');
                if(pos != std::string::npos) {
                        opt = arg.substr(0, pos);
                        optValue = arg.substr(pos + 1);
                }

                ArgType t;
                if(get(opt, t)) {
                        // It is an option

                        OptArg a;

                        if(t.hasArg) {
                                // It requires an argument

                                if(pos == std::string::npos) {
                                        if(i == argc - 1) {
                                                throw std::string("'" + opt + "' requires an argument");
                                        }

                                        optValue = std::string(argv[++i]);
                                }

                                a.option = opt;
                                a.arg = optValue;

                        } else {
                                // It does not require an argument

                                a.option = opt;
                                a.arg = "";
                        }

                        _optArgs.push_back(a);

                } else {
                        // It is an operand

                        _operands.push_back(arg);
                }
        }
}

std::vector<OptArg> CmdParse::getArgs()
{
	return _optArgs;
}

std::vector<std::string> CmdParse::getOperands()
{
	return _operands;
}