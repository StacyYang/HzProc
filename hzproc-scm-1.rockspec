package = "hzproc"
version = "scm-1"

source = {
		url = "git@github.com:zhanghang1989/hzproc.git",
		tag = "master"
}

description = {
		summary = "An image processing toolbox for torch packages",
		detailed = [[
					A torch image processing toolbox for data augmentation
					]],
		homepage = "https://github.com/zhanghang1989/hzproc"
}

dependencies = {
		"torch >= 7.0",
		"cutorch >= 1.0"
}

build = {
   type = "cmake",
   variables = {
      CMAKE_BUILD_TYPE="Release",
      CMAKE_PREFIX_PATH="$(LUA_BINDIR)/..",
      CMAKE_INSTALL_PREFIX="$(PREFIX)"
   }
}
