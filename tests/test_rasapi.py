# This file is part of rasapi.

# rasapi is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# rasapi is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with rasapi.  If not, see <https://www.gnu.org/licenses/>.

# Copyright 2019, Armin Schaare <armin-scha@hotmail.de>

import pytest
from unittest.mock import MagicMock

import requests

from rasapi.rasapi import RasaPI, Event


class MockResponse:
    def __init__(self, json_data={}, status_code=200):
        self.json_data = json_data
        self.status_code = status_code
        self.text = 'OK'

    def json(self):
        return self.json_data

    @property
    def ok(self):
        return self.status_code < 400


def get_mock_response(json_data={}, status_code=200):
    req_mock = requests.sessions.Session.request = \
        MagicMock(return_value=MockResponse(json_data, status_code))
    return req_mock


class TestRasaPI:
    # @pytest.fixture(autouse=True)
    # def no_requests(self, request):
    #     self.req_mock = requests.sessions.Session.request = \
    #         MagicMock(return_value=get_mock_response(request.function.__name__))

    @pytest.fixture
    def rpi(self):
        return RasaPI(url='http://nowhere/', token='')

    def test_is_healthy(self, rpi):
        req_mock = get_mock_response()
        assert rpi.is_healthy
        req_mock.assert_called_once_with(method='GET', url='http://nowhere/')

    def test_version(self, rpi):
        req_mock = get_mock_response(json_data={'version': '1.0.1'})
        assert rpi.version == '1.0.1'
        req_mock.assert_called_once_with(
            method='GET', url='http://nowhere/version'
        )

    def test_minimum_compatible_version(self, rpi):
        req_mock = get_mock_response(
            json_data={'minimum_compatible_version': '1.0.0rc1'}
        )
        assert rpi.minimum_compatible_version == '1.0.0rc1'
        req_mock.assert_called_once_with(
            method='GET', url='http://nowhere/version'
        )

    def test_status(self, rpi):
        req_mock = get_mock_response()
        rpi.status
        req_mock.assert_called_once_with(
            method='GET', url='http://nowhere/status'
        )

    def test_get_tracker(self, rpi):
        req_mock = get_mock_response()
        rpi.get_tracker('12345678')
        req_mock.assert_called_once_with(
            method='GET', url='http://nowhere/conversations/12345678/tracker'
        )

    def test_append_event(self, rpi):
        req_mock = get_mock_response()
        rpi.append_event('12345678', 'slot')
        req_mock.assert_called_once_with(
            method='POST',
            url='http://nowhere/conversations/12345678/tracker/events',
            json={'event': 'slot'}
        )

    def test_append_events(self, rpi):
        req_mock = get_mock_response()
        events = [Event('slot'), Event('wurst', 1234), Event('calzone', 4321)]
        rpi.append_events('12345678', events)
        req_mock.assert_called_once_with(
            method='POST',
            url='http://nowhere/conversations/12345678/tracker/events',
            json=[
                {'event': 'slot'},
                {'event': 'wurst', 'timestamp': 1234},
                {'event': 'calzone', 'timestamp': 4321}
            ]
        )

    def test_replace_events(self, rpi):
        req_mock = get_mock_response()
        events = [Event('slot'), Event('wurst', 1234), Event('calzone', 4321)]
        rpi.replace_events('12345678', events)
        req_mock.assert_called_once_with(
            method='PUT',
            url='http://nowhere/conversations/12345678/tracker/events',
            json=[
                {'event': 'slot'},
                {'event': 'wurst', 'timestamp': 1234},
                {'event': 'calzone', 'timestamp': 4321}
            ]
        )

    def test_get_story(self, rpi):
        req_mock = get_mock_response()
        rpi.get_story('12345678')
        req_mock.assert_called_once_with(
            method='GET',
            url='http://nowhere/conversations/12345678/story'
        )

    def test_execute_action(self, rpi):
        req_mock = get_mock_response()
        rpi.execute_action(
            '12345678', 'utter_test', policy='string', confidence=1.0)
        req_mock.assert_called_once_with(
            method='POST',
            url='http://nowhere/conversations/12345678/execute',
            params={'include_events': 'AFTER_RESTART'},
            json={'name': 'utter_test', 'policy': 'string', 'confidence': 1.0}
        )

    def test_score_actions(self, rpi):
        req_mock = get_mock_response()
        rpi.score_actions('12345678')
        req_mock.assert_called_once_with(
            method='POST',
            url='http://nowhere/conversations/12345678/predict'
        )

    def test_add_message(self, rpi):
        req_mock = get_mock_response()
        rpi.add_message('12345678', 'hello', 'testuser', parse_data={})
        req_mock.assert_called_once_with(
            method='POST',
            url='http://nowhere/conversations/12345678/messages',
            params={'include_events': 'AFTER_RESTART'},
            json={'text': 'hello', 'sender': 'user', 'parse_data': {}}
        )

    def test_train_model(self, rpi):
        req_mock = get_mock_response()
        rpi.train_model(
            'wurst', domain='a', nlu='b', stories='c', out='d', force=False)
        req_mock.assert_called_once_with(
            method='POST',
            url='http://nowhere/model/train',
            json={'config': 'wurst', 'domain': 'a', 'nlu': 'b',
                  'stories': 'c', 'out': 'd', 'force': False}
        )

    def test_evaluate_stories(self, rpi):
        req_mock = get_mock_response()
        rpi.evaluate_stories("MARKDOWN")
        req_mock.assert_called_once_with(
            method='POST',
            url='http://nowhere/model/test/stories',
            params={'e2e': 'false'},
            data="MARKDOWN"
        )

    def test_evaluate_intents(self, rpi):
        req_mock = get_mock_response()
        rpi.evaluate_intents("MARKDOWN", "model.tar.gz")
        req_mock.assert_called_once_with(
            method='POST',
            url='http://nowhere/model/test/intents',
            params={'model': 'model.tar.gz'},
            data="MARKDOWN"
        )

    def test_predict_action(self, rpi):
        req_mock = get_mock_response()
        rpi.predict_action(['slot', 'action', 'whut'])
        req_mock.assert_called_once_with(
            method='POST',
            url='http://nowhere/model/predict',
            params={'include_events': 'AFTER_RESTART'},
            json=[{'event': 'slot'}, {'event': 'action'}, {'event': 'whut'}]
        )

    def test_parse_message(self, rpi):
        req_mock = get_mock_response()
        rpi.parse_message('message')
        req_mock.assert_called_once_with(
            method='POST',
            url='http://nowhere/model/parse',
            params={'emulation_mode': 'LUIS'},
            json={'text': 'message'}
        )

    def test_replace_current_model(self, rpi):
        req_mock = get_mock_response()
        rpi.replace_current_model(
            '/models/model.tar.gz',
            {'url': 'nowhere', 'params': {}},
            'aws'
        )
        req_mock.assert_called_once_with(
            method='PUT',
            url='http://nowhere/model',
            json={
                'model_file': '/models/model.tar.gz',
                'model_server': {'url': 'nowhere', 'params': {}},
                'remote_storage': 'aws'
            }
        )

    def test_unload_current_model(self, rpi):
        req_mock = get_mock_response()
        rpi.unload_current_model()
        req_mock.assert_called_once_with(
            method='DELETE', url='http://nowhere/model'
        )

    def test_domain(self, rpi):
        req_mock = get_mock_response()
        _ = rpi.domain()
        req_mock.assert_called_once_with(
            method='GET', url='http://nowhere/domain',
            headers={"Accept": "application/yaml"}
        )
