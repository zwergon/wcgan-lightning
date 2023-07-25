import unittest

from gan.generator import Generator

import matplotlib.pyplot as plt


class TestGenerator(unittest.TestCase):


    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.gen = Generator()
        self.num_test = 100

    def test_noise(self):
        test_hidden_noise = Generator.get_noise(2, self.gen.z_dim)
        print(test_hidden_noise)
        test_uns_noise = self.gen.unsqueeze_noise(test_hidden_noise)
        print(test_uns_noise.shape)
        gen_output = self.gen(test_uns_noise)
        plt.imshow(gen_output.detach().numpy()[0, 0, :, :])
        plt.show()
        plt.imshow(gen_output.detach().numpy()[1, 0, :, :])
        plt.show()

    def test_hidden_block(self):
        test_hidden_noise = Generator.get_noise(self.num_test, self.gen.z_dim)
        test_hidden_block = self.gen.make_gen_block(10, 20, kernel_size=4, stride=1)
        test_uns_noise = self.gen.unsqueeze_noise(test_hidden_noise)
        hidden_output = test_hidden_block(test_uns_noise)
        self.assertEqual(tuple(hidden_output.shape), (self.num_test, 20, 4, 4))
        self.assertTrue( hidden_output.max() > 1 )
        self.assertEqual(hidden_output.min(), 0 )
        self.assertTrue( hidden_output.std() > 0.2 )
        self.assertTrue( hidden_output.std() < 1 )
        self.assertTrue( hidden_output.std() > 0.5 )

    def test_final(self):
        test_final_noise = Generator.get_noise(self.num_test, self.gen.z_dim) * 20
        test_final_block = self.gen.make_gen_block(10, 20, final_layer=True)
        test_final_uns_noise = self.gen.unsqueeze_noise(test_final_noise)
        final_output = test_final_block(test_final_uns_noise)
        self.assertEqual(final_output.max().item(), 1)
        self.assertEqual(final_output.min().item(), -1)

    def test_whole_thing(self):
        test_gen_noise = Generator.get_noise(self.num_test, self.gen.z_dim)
        test_uns_gen_noise = self.gen.unsqueeze_noise(test_gen_noise)
        gen_output = self.gen(test_uns_gen_noise)
        self.assertEqual(tuple(gen_output.shape), (self.num_test, 1, 28, 28))
        self.assertTrue(gen_output.std() > .5)
        self.assertTrue(gen_output.std() < .8)

       
            


if __name__ == "__main__":
    unittest.main()