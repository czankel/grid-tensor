//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <iostream>
#include <memory>
#include <string>

#include <unistd.h>

#include <grid/models/llama.h>
#include <grid/tensor/tensor_base.h>
#include <grid/util/demangle.h>

int main(int argc, char** argv)
{
  int                   opt;
  std::string           model_path;
  grid::LLaMAFile::Type model_type = grid::LLaMAFile::kGgml;

  int                   steps = 256;
  bool                  show_info = false;

  while ((opt = getopt(argc, argv, "vhim:t:")) != -1)
  {
    switch (opt)
    {
      case 'h': // help
        // FIXME: TODO
        std::cout << "Help : TODO" << std::endl;
        exit(0);

      case 'i': // info
        show_info = true;
        break;

      case 'v': // version
        // FIXME: TODO
        std::cout << "Version: " << std::endl;
        break;

      case 'm': // model file
        model_path = optarg;
        break;

      case 't': // file type/format
        std::string type(optarg);
        if (type == "karpathy")
          model_type = grid::LLaMAFile::kKarpathy;
        break;
    }
  }

  if (model_path.empty())
  {
      std::cerr << "no model provided" << std::endl;
      exit(1);
  }

  std::unique_ptr<grid::LLaMAFile> file;
  try
  {
    file.reset(grid::LLaMAFile::Open(model_type, model_path));
  }
  catch(std::string err)
  {
    std::cerr << "Error: " << err << std::endl;
    exit(1);
  }

  file->PrintModelInfo(std::cout);


  std::unique_ptr<grid::LLaMAModel> model(grid::LLaMAModel::Load<grid::Tensor>(*file));
  if (show_info)
    return 0;

  // take extra arguments as text input; concatenate with a space.
  std::string prompt;
  for ( ; optind < argc; optind++)
    prompt.append(argv[optind]).append(1, ' ');
  prompt.resize(prompt.size() - 1);

  std::cout << "PROMPT " << prompt << std::endl;

  model->Predict(prompt, steps);

  return 0;
}
