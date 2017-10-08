def profile(func):
	def wrap(*args, **kwargs):
		print('Function:', func.__name__)
		started_at = time.time()
		result = func(*args, **kwargs)
		print('time:', time.time() - started_at)
		return result

	return wrap

