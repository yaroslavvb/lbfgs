function saveTensor0D(tensor, fname)
   local out = assert(io.open(fname, "w")) -- open a file for serialization
   out:write(tensor)
   out:write("\n")
   out:close()
end


function saveTensor1D(tensor, fname)
   local out = assert(io.open(fname, "w")) -- open a file for serialization

   for i=1,tensor:size(1) do
      out:write(tensor[i])
      out:write("\n")
   end
   out:close()
end

