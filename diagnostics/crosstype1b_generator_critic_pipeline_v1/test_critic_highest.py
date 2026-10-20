import jax
from test_critic_model import main
with jax.default_matmul_precision('highest'):
    main()
