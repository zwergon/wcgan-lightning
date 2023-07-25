import unittest

from gan.generator import Generator
from gan.discriminator import Discriminator

class TestDiscriminator(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

        self.num_test = 100

        gen = Generator()
        self.disc = Discriminator()
        self.test_images = gen(Generator.get_noise(self.num_test, gen.z_dim))


    def test_hidden_block(self):
        test_hidden_block = self.disc.make_disc_block(1, 5, kernel_size=6, stride=3)
        hidden_output = test_hidden_block(self.test_images)
        self.assertEqual( tuple(hidden_output.shape), (self.num_test, 5, 8, 8))
        # Because of the LeakyReLU slope
        self.assertTrue( -hidden_output.min() / hidden_output.max() > 0.15 )
        self.assertTrue( -hidden_output.min() / hidden_output.max() < 0.25 )
        self.assertTrue( hidden_output.std() > 0.5 )
        self.assertTrue( hidden_output.std() < 1 )

    def test_final_block(self):
        test_final_block = self.disc.make_disc_block(1, 10, kernel_size=2, stride=5, final_layer=True)
        final_output = test_final_block(self.test_images)
        
        self.assertEqual( tuple(final_output.shape), (self.num_test, 10, 6, 6) )
        self.assertTrue( final_output.max() > 1.0 )
        self.assertTrue( final_output.min() < -1.0 )
        self.assertTrue( final_output.std() > 0.3 )
        self.assertTrue( final_output.std() < 0.6 )

    def test_whole_thing(self):
        disc_output = self.disc(self.test_images)
                
        self.assertEqual( tuple(disc_output.shape),  (self.num_test, 1) )
        self.assertTrue( disc_output.std() > 0.25 )
        self.assertTrue( disc_output.std() < 0.5 )


if __name__ == "__main__":
    unittest.main()