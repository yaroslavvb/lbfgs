require 'torch'
require 'paths'

mnist = {}

mnist.path_remote = 'https://s3.amazonaws.com/torch7/data/mnist.t7.tgz'
mnist.path_dataset = 'mnist.t7'
mnist.path_trainset = paths.concat(mnist.path_dataset, 'train_32x32.t7')
--mnist.path_trainset = paths.concat(mnist.path_dataset, 'train_28x28.t7')
mnist.path_testset = paths.concat(mnist.path_dataset, 'test_32x32.t7')
--mnist.path_testset = paths.concat(mnist.path_dataset, 'test_28x28.t7')

function mnist.download()
   if not paths.filep(mnist.path_trainset) or not paths.filep(mnist.path_testset) then
      local remote = mnist.path_remote
      local tar = paths.basename(remote)
      os.execute('wget ' .. remote .. '; ' .. 'tar xvf ' .. tar .. '; rm ' .. tar)
   end
end

function mnist.loadTrainSet(maxLoad, geometry)
   return mnist.loadDataset(mnist.path_trainset, maxLoad, geometry)
end

function mnist.saveTrainSet()
   return mnist.saveDataset(mnist.path_trainset, mnist.path_trainset .. ".csv")
end

function mnist.loadTestSet(maxLoad, geometry)
   return mnist.loadDataset(mnist.path_testset, maxLoad, geometry)
end

function mnist.saveTestSet()
   return mnist.saveDataset(mnist.path_testset, mnist.path_testset .. ".csv")
end

function serializeLabels(tensor, fname)
   local out = assert(io.open(fname, "w")) -- open a file for serialization

   splitter = ","
   for i=1,tensor:size(1) do
      out:write(tensor[i])
      out:write("\n")
   end
   out:close()
end

function serializeTensor(tensor)
   local out = assert(io.open("./dump.csv", "w")) -- open a file for serialization

   splitter = ","
   for i=1,tensor:size(1) do
      for j=1,tensor:size(2) do
	 out:write(tensor[i][j])
	 if j == tensor:size(2) then
            out:write("\n")
	 else
            out:write(splitter)
	 end
      end
   end
   out:close()
end

function serializeData(tensor, fname)
   local out = assert(io.open(fname, "w")) -- open a file for serialization

   for i=1,tensor:size(1) do
      local j = 1
      for k=1,tensor:size(3) do
	 for l=1,tensor:size(4) do
	    out:write(tensor[i][j][k][l])
	    out:write(",")
	 end
	 out:write("\n")
      end
      --      out:write("---\n")
   end
   out:close()
end


function mnist.saveDataset(fileName, outprefix)
   local f = torch.load(fileName, 'ascii')
   local data = f.data
   --:type(torch.getdefaulttensortype())
   local labels = f.labels
   local labels_fname = outprefix..".labels"
   print("Saving dataset labels ", fileName, " to ", labels_fname)
   serializeLabels(labels, labels_fname)
   local data_fname = outprefix..".data"
   print("Saving dataset data ", fileName, " to ", data_fname)
   serializeData(data, data_fname)

end
   
function mnist.loadDataset(fileName, maxLoad)
   mnist.download()
   print("Loading dataset", fileName)
   local f = torch.load(fileName, 'ascii')
   local data = f.data:type(torch.getdefaulttensortype())
   local labels = f.labels

   
   local nExample = f.data:size(1)
   if maxLoad and maxLoad > 0 and maxLoad < nExample then
      nExample = maxLoad
      print('<mnist> loading only ' .. nExample .. ' examples')
   end
   data = data[{{1,nExample},{},{},{}}]
   labels = labels[{{1,nExample}}]
   print('<mnist> done')

   local dataset = {}
   dataset.data = data
   dataset.labels = labels

   function dataset:normalize(mean_, std_)
      local mean = mean_ or data:view(data:size(1), -1):mean(1)
      local std = std_ or data:view(data:size(1), -1):std(1, true)
      for i=1,data:size(1) do
         data[i]:add(-mean[1][i])
         if std[1][i] > 0 then
            tensor:select(2, i):mul(1/std[1][i])
         end
      end
      return mean, std
   end

   function dataset:normalizeGlobal(mean_, std_)
      local std = std_ or data:std()
      local mean = mean_ or data:mean()
      data:add(-mean)
      data:mul(1/std)
      return mean, std
   end

   function dataset:size()
      return nExample
   end

   local labelvector = torch.zeros(10)

   setmetatable(dataset, {__index = function(self, index)
			     local input = self.data[index]
			     local class = self.labels[index]
			     local label = labelvector:zero()
			     label[class] = 1
			     local example = {input, label}
                                       return example
   end})

   return dataset
end
