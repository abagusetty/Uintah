<start>
<upsFile>RMCRT_test_1L.ups</upsFile>
<gnuplot>
  <script>plotScript.gp</script>s
  <title>RMCRT order-of-accuracy \\n 1 timestep (41^3)</title>
  <ylabel>Error</ylabel>
  <xlabel># of Rays</xlabel>
</gnuplot>

<AllTests>
  <replace_lines>
    <max_Timesteps>1</max_Timesteps>
    <randomSeed> true </randomSeed>
  </replace_lines>
</AllTests>

<Test>
    <Title>8</Title>
    <sus_cmd> sus </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 1</postProcess_cmd>
    <x>8</x>
    <replace_lines>
      <NoOfRays>          8        </NoOfRays>
    </replace_lines>
</Test>

<Test>
    <Title>16</Title>
    <sus_cmd> sus </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper  -bm 1</postProcess_cmd>
    <x>16</x>
    <replace_lines>
      <NoOfRays>          16        </NoOfRays>
    </replace_lines>
</Test>

<Test>
    <Title>32</Title>
    <sus_cmd> sus </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 1</postProcess_cmd>
    <x>32</x>
    <replace_lines>
      <NoOfRays>          32        </NoOfRays>
    </replace_lines>
</Test>

<Test>
    <Title>64</Title>
    <sus_cmd> sus </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 1</postProcess_cmd>
    <x>64</x>
    <replace_lines>
      <NoOfRays>          64        </NoOfRays>
    </replace_lines>
</Test>

<Test>
    <Title>128</Title>
    <sus_cmd> sus </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 1</postProcess_cmd>
    <x>128</x>
    <replace_lines>
      <NoOfRays>          128        </NoOfRays>
    </replace_lines>
</Test>

</start>
