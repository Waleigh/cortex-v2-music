import asyncio
import json
from lib.cortex import Cortex


async def do_stuff(cortex):
    # await cortex.inspectApi()
    print("** USER LOGIN **")
    await cortex.get_user_login()
    print("** GET CORTEX INFO **")
    await cortex.get_cortex_info()
    print("** HAS ACCESS RIGHT **")
    await cortex.has_access_right()
    print("** REQUEST ACCESS **")
    await cortex.request_access()
    print("** AUTHORIZE **")
    await cortex.authorize(debit=10000)
    print("** GET LICENSE INFO **")
    await cortex.get_license_info()
    print("** QUERY HEADSETS **")
    await cortex.query_headsets()
    if len(cortex.headsets) > 0:
        print("** CREATE SESSION **")
        await cortex.create_session(activate=True,
                                    headset_id=cortex.headsets[0])
        print("** CREATE RECORD **")
        await cortex.create_record(title="test record 1")
        print("** SUBSCRIBE MET **")
        await cortex.subscribe(['met'])
        while input().strip() is not "q":
            data = json.loads(await cortex.get_data())
            print("Data:", data)
            print("Met:", data['met'])
            print("Engagement:", data["met"][1])
            print("Excitement:", data["met"][3])
            print("Long-term Excitement:", data["met"][4])
            print("Stress:", data["met"][6])
            print("Relaxation:", data["met"][8])
            print("Interest:", data["met"][10])
            print("Focus:", data["met"][12])
        await cortex.close_session()


def run():
    cortex = Cortex('./cortex_creds')
    asyncio.run(do_stuff(cortex))
    cortex.close()


if __name__ == '__main__':
    run()
