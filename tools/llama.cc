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
  enum Type
  {
    kKarpathy,    ///> https://github.com/karpathy/llama2.c
    kGgml,
  };

  int         opt;
  std::string model_path;
  Type        model_type = kGgml;

  int         steps = 256;

  while ((opt = getopt(argc, argv, "vhm:t:")) != -1)
  {
    switch (opt)
    {
      case 'h': // help
        // FIXME: TODO
        std::cout << "Help : TODO" << std::endl;
        exit(0);

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
          model_type = kKarpathy;
        break;

        // FIXME override, steps, etc.
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
    switch (model_type)
    {
      // FIXME: should fail if invalid file
      case kKarpathy: file.reset(new grid::KarpathyFile(model_path)); break;
      default: break;
    }
  }
  catch(std::string err)
  {
    std::cerr << "Error: " << err << std::endl;
    exit(1);
  }

  // if file info, print info FIXME: crashes when invalid -t tokenfile?????? no file??
  file->PrintModelInfo(std::cout);
  std::unique_ptr<grid::LLaMAModel> model(grid::LLaMAModel::Load<grid::Tensor>(*file));

  // take extra arguments as text input; concatenate with a space.
  std::string prompt;
  for ( ; optind < argc; optind++)
    prompt.append(argv[optind]).append(1, ' ');
  prompt.resize(prompt.size() - 1);

  model->Generate(prompt, steps);

  return 0;
}
