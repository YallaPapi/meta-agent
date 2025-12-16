"""
Appium UI Controller - encapsulates all Appium-based UI interactions.

This module provides a clean interface for interacting with Android
UI elements through Appium WebDriver.

Extended with DM automation methods for Instagram messaging.
"""
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

from appium import webdriver
from appium.webdriver.common.appiumby import AppiumBy


@dataclass
class ScreenGeometry:
    """Screen dimensions and common coordinates.

    Injectable configuration for screen-size-dependent operations.
    Default values assume 720x1280 resolution (common for cloud phones).
    """
    width: int = 720
    height: int = 1280

    # Swipe durations (ms)
    swipe_duration_fast: int = 200
    swipe_duration_slow: int = 500

    @property
    def center_x(self) -> int:
        return self.width // 2

    @property
    def center_y(self) -> int:
        return self.height // 2

    @property
    def feed_top_y(self) -> int:
        """Top of scrollable content area."""
        return int(self.height * 0.3)

    @property
    def feed_bottom_y(self) -> int:
        """Bottom of scrollable content area."""
        return int(self.height * 0.7)

    @property
    def message_input_y(self) -> int:
        """Typical Y position of message input field."""
        return int(self.height * 0.9)

    @property
    def dm_button_x(self) -> int:
        """Typical X position of DM button (top right)."""
        return int(self.width * 0.94)

    @property
    def dm_button_y(self) -> int:
        """Typical Y position of DM button."""
        return int(self.height * 0.09)


# Default geometry for backward compatibility
DEFAULT_SCREEN = ScreenGeometry()


class AppiumUIController:
    """Controls Android UI through Appium WebDriver."""

    # Instagram element identifiers
    DM_INBOX_BUTTON_DESC = "Direct message inbox"
    DM_INBOX_BUTTON_ID = "com.instagram.android:id/action_bar_inbox_button"
    MESSAGE_INPUT_ID = "com.instagram.android:id/row_thread_composer_edittext"
    SEND_BUTTON_DESC = "Send"
    CAMERA_BUTTON_DESC = "Camera"
    GALLERY_BUTTON_DESC = "Gallery"

    def __init__(self, driver: webdriver.Remote, screen: ScreenGeometry = None):
        """
        Initialize the controller.

        Args:
            driver: Appium WebDriver instance (must already be connected).
            screen: Screen geometry configuration (uses defaults if not provided).
        """
        self._driver = driver
        self._screen = screen or DEFAULT_SCREEN

    @property
    def driver(self) -> webdriver.Remote:
        """Get the underlying Appium driver."""
        return self._driver

    def tap(self, x: int, y: int, delay: float = 1.5) -> None:
        """Tap at coordinates.

        Args:
            x: X coordinate.
            y: Y coordinate.
            delay: Delay after tap in seconds.
        """
        print(f"  [TAP] ({x}, {y})")
        if not self._driver:
            raise Exception("Appium driver not connected - cannot tap")
        self._driver.tap([(x, y)])
        time.sleep(delay)

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> None:
        """Swipe from one point to another.

        Args:
            x1: Start X coordinate.
            y1: Start Y coordinate.
            x2: End X coordinate.
            y2: End Y coordinate.
            duration_ms: Duration of swipe in milliseconds.
        """
        if not self._driver:
            raise Exception("Appium driver not connected - cannot swipe")
        self._driver.swipe(x1, y1, x2, y2, duration_ms)

    def press_key(self, keycode) -> None:
        """Press a key.

        Args:
            keycode: Key code (int) or string like 'KEYCODE_BACK'.
        """
        if not self._driver:
            raise Exception("Appium driver not connected - cannot press key")

        key_map = {
            'KEYCODE_BACK': 4,
            'KEYCODE_HOME': 3,
            'KEYCODE_ENTER': 66,
        }

        if isinstance(keycode, str):
            keycode = key_map.get(keycode, 4)  # Default to BACK

        self._driver.press_keycode(keycode)

    def type_text(self, text: str) -> bool:
        """Type text into the currently focused field.

        Args:
            text: Text to type (supports Unicode/emojis).

        Returns:
            True if text was typed successfully.
        """
        if not self._driver:
            print("    ERROR: Appium driver not connected!")
            return False

        print(f"    Typing via Appium ({len(text)} chars)...")
        try:
            # Find the currently focused EditText element
            edit_texts = self._driver.find_elements(AppiumBy.CLASS_NAME, "android.widget.EditText")
            if edit_texts:
                for et in edit_texts:
                    if et.is_displayed():
                        et.send_keys(text)
                        print("    Appium: text sent successfully")
                        time.sleep(0.8)
                        return True

            # Fallback: try to type using the active element
            active = self._driver.switch_to.active_element
            if active:
                active.send_keys(text)
                print("    Appium: text sent to active element")
                time.sleep(0.8)
                return True

            print("    ERROR: No text field found to type into")
            return False

        except Exception as e:
            print(f"    Appium typing error: {e}")
            return False

    def dump_ui(self) -> Tuple[List[Dict], str]:
        """Dump UI hierarchy and return parsed elements.

        Returns:
            Tuple of (elements list, raw XML string).
            Elements have: text, desc, id, bounds, center, clickable.

        Raises:
            Exception: If driver not connected or dump fails.
        """
        elements = []
        xml_str = ""

        if not self._driver:
            raise Exception("Appium driver not connected - cannot dump UI")

        xml_str = self._driver.page_source

        if '<?xml' not in xml_str:
            return elements, xml_str

        xml_clean = xml_str[xml_str.find('<?xml'):]
        try:
            root = ET.fromstring(xml_clean)
            # Appium uses class names as tags, iterate over ALL elements
            for elem in root.iter():
                text = elem.get('text', '')
                desc = elem.get('content-desc', '')
                res_id = elem.get('resource-id', '')
                bounds = elem.get('bounds', '')
                clickable = elem.get('clickable', 'false')

                if bounds and (text or desc or clickable == 'true'):
                    # Parse bounds [x1,y1][x2,y2]
                    m = re.match(r'\[(\d+),(\d+)\]\[(\d+),(\d+)\]', bounds)
                    if m:
                        x1, y1, x2, y2 = map(int, m.groups())
                        cx, cy = (x1+x2)//2, (y1+y2)//2
                        elements.append({
                            'text': text,
                            'desc': desc,
                            'id': res_id.split('/')[-1] if '/' in res_id else res_id,
                            'bounds': bounds,
                            'center': (cx, cy),
                            'clickable': clickable == 'true'
                        })
        except ET.ParseError as e:
            print(f"  XML parse error: {e}")

        return elements, xml_str

    def find_element_by_text(self, text: str, partial: bool = False) -> Optional[Dict]:
        """Find an element by its text content.

        Args:
            text: Text to search for.
            partial: If True, match partial text.

        Returns:
            Element dict with center coordinates, or None.
        """
        elements, _ = self.dump_ui()
        for elem in elements:
            elem_text = elem.get('text', '')
            elem_desc = elem.get('desc', '')
            if partial:
                if text.lower() in elem_text.lower() or text.lower() in elem_desc.lower():
                    return elem
            else:
                if elem_text == text or elem_desc == text:
                    return elem
        return None

    def find_element_by_id(self, element_id: str) -> Optional[Dict]:
        """Find an element by its resource ID.

        Args:
            element_id: Resource ID (without package prefix).

        Returns:
            Element dict with center coordinates, or None.
        """
        elements, _ = self.dump_ui()
        for elem in elements:
            if elem.get('id', '') == element_id or element_id in elem.get('id', ''):
                return elem
        return None

    def tap_element(self, element: Dict, delay: float = 1.5) -> None:
        """Tap on an element by its center coordinates.

        Args:
            element: Element dict with 'center' key.
            delay: Delay after tap.
        """
        center = element.get('center')
        if center:
            self.tap(center[0], center[1], delay)
        else:
            raise ValueError("Element has no center coordinates")

    def is_keyboard_visible(self, adb_shell_func=None) -> bool:
        """Check if the keyboard is currently visible.

        Args:
            adb_shell_func: Optional function to run ADB shell commands.
                           If not provided, returns False.

        Returns:
            True if keyboard is visible.
        """
        if not adb_shell_func:
            return False

        # Method 1: Check dumpsys for keyboard visibility
        result = adb_shell_func("dumpsys input_method | grep mInputShown")
        if "mInputShown=true" in result:
            return True

        # Method 2: Check window visibility
        result = adb_shell_func("dumpsys window | grep -i keyboard")
        if "isVisible=true" in result.lower() or "mhasfocus=true" in result.lower():
            return True

        # Method 3: Check if InputMethod window is visible
        result = adb_shell_func("dumpsys window windows | grep -E 'mCurrentFocus|mFocusedApp'")
        if "InputMethod" in result:
            return True

        return False

    def save_screenshot(self, filepath: str) -> bool:
        """Save a screenshot to file.

        Args:
            filepath: Path to save screenshot.

        Returns:
            True if screenshot was saved.
        """
        try:
            if self._driver:
                self._driver.save_screenshot(filepath)
                return True
        except Exception as e:
            print(f"    Failed to save screenshot: {e}")
        return False

    def scroll_down(self) -> None:
        """Scroll down on the screen."""
        self.swipe(
            self._screen.center_x, self._screen.feed_bottom_y,
            self._screen.center_x, self._screen.feed_top_y,
            self._screen.swipe_duration_fast
        )

    def scroll_up(self) -> None:
        """Scroll up on the screen."""
        self.swipe(
            self._screen.center_x, self._screen.feed_top_y,
            self._screen.center_x, self._screen.feed_bottom_y,
            self._screen.swipe_duration_fast
        )

    def go_back(self) -> None:
        """Press the back button."""
        self.press_key('KEYCODE_BACK')

    def go_home(self) -> None:
        """Press the home button."""
        self.press_key('KEYCODE_HOME')

    # =========================================================================
    # DM AUTOMATION METHODS
    # =========================================================================

    def open_dm_inbox(self) -> bool:
        """Navigate to Instagram DM inbox.

        Returns:
            True if inbox was opened successfully.
        """
        print("  [DM] Opening DM inbox...")

        # Try finding by content-desc first
        elem = self.find_element_by_text(self.DM_INBOX_BUTTON_DESC)
        if elem:
            self.tap_element(elem)
            time.sleep(2)
            return True

        # Try finding by resource ID
        elem = self.find_element_by_id("action_bar_inbox_button")
        if elem:
            self.tap_element(elem)
            time.sleep(2)
            return True

        # Fallback: tap typical DM button location (top right)
        print("    DM button not found, trying typical location...")
        self.tap(self._screen.dm_button_x, self._screen.dm_button_y, delay=2)
        return True

    def get_dm_conversations(self, limit: int = 10) -> List[Dict]:
        """Get list of DM conversations from inbox.

        Args:
            limit: Maximum conversations to return.

        Returns:
            List of conversation dicts with user_id, username, preview, unread.
        """
        conversations = []
        elements, _ = self.dump_ui()

        # Look for conversation list items
        # Instagram uses RecyclerView with conversation rows
        for elem in elements:
            text = elem.get('text', '')
            desc = elem.get('desc', '')

            # Skip non-conversation elements
            if not elem.get('clickable'):
                continue

            # Look for conversation preview patterns
            # Typically format: "username\nLast message preview"
            if '\n' in text or desc:
                parts = (text or desc).split('\n')
                if len(parts) >= 1:
                    username = parts[0].strip()
                    preview = parts[1].strip() if len(parts) > 1 else ""

                    # Skip system items
                    if username.lower() in ['search', 'requests', 'edit']:
                        continue

                    conversations.append({
                        'username': username,
                        'preview': preview,
                        'center': elem.get('center'),
                        'unread': 'unread' in desc.lower() or 'unseen' in desc.lower(),
                    })

                    if len(conversations) >= limit:
                        break

        return conversations

    def get_unread_dms(self) -> List[Dict]:
        """Get list of unread DM conversations.

        Returns:
            List of unread conversation dicts.
        """
        all_convos = self.get_dm_conversations(limit=20)
        return [c for c in all_convos if c.get('unread')]

    def open_conversation(self, username: str) -> bool:
        """Open a specific DM conversation by username.

        Args:
            username: Instagram username to open chat with.

        Returns:
            True if conversation was opened.
        """
        print(f"  [DM] Opening conversation with {username}...")

        # First ensure we're in inbox
        elem = self.find_element_by_text(username, partial=True)
        if elem:
            self.tap_element(elem)
            time.sleep(2)
            return True

        # Try scrolling to find
        for _ in range(3):
            self.scroll_down()
            time.sleep(1)
            elem = self.find_element_by_text(username, partial=True)
            if elem:
                self.tap_element(elem)
                time.sleep(2)
                return True

        print(f"    Could not find conversation with {username}")
        return False

    def read_conversation_messages(self, limit: int = 10) -> List[Dict]:
        """Read messages from current open conversation.

        Args:
            limit: Maximum messages to return (most recent).

        Returns:
            List of message dicts with role, content, timestamp.
        """
        messages = []
        elements, _ = self.dump_ui()

        # Messages are typically in a RecyclerView
        # User messages vs bot messages have different alignments
        for elem in elements:
            text = elem.get('text', '').strip()
            desc = elem.get('desc', '')

            if not text:
                continue

            # Skip UI elements
            if text.lower() in ['send', 'message...', 'camera', 'gallery', 'gif']:
                continue

            # Determine if sent or received based on position
            # Instagram: received = left aligned, sent = right aligned
            center_x = elem.get('center', (0, 0))[0]
            is_received = center_x < self._screen.center_x

            messages.append({
                'role': 'user' if is_received else 'assistant',
                'content': text,
                'center': elem.get('center'),
            })

        # Return most recent
        return messages[-limit:] if len(messages) > limit else messages

    def get_latest_message(self) -> Optional[Dict]:
        """Get the latest message in current conversation.

        Returns:
            Message dict or None.
        """
        messages = self.read_conversation_messages(limit=1)
        return messages[-1] if messages else None

    def send_dm_message(self, text: str) -> bool:
        """Send a text message in current DM conversation.

        Args:
            text: Message text to send.

        Returns:
            True if message was sent.
        """
        print(f"  [DM] Sending message: {text[:50]}...")

        # Find message input field
        input_elem = self.find_element_by_id("row_thread_composer_edittext")
        if not input_elem:
            input_elem = self.find_element_by_text("Message...", partial=True)

        if input_elem:
            self.tap_element(input_elem, delay=0.5)
        else:
            # Tap typical message input location (bottom center)
            self.tap(self._screen.center_x, self._screen.message_input_y, delay=0.5)

        # Type the message
        if not self.type_text(text):
            print("    Failed to type message")
            return False

        time.sleep(0.5)

        # Find and tap send button
        send_elem = self.find_element_by_text("Send")
        if send_elem:
            self.tap_element(send_elem)
        else:
            # Try send button by ID
            send_elem = self.find_element_by_id("send_button")
            if send_elem:
                self.tap_element(send_elem)
            else:
                # Tap typical send button location (right side at message input level)
                send_x = int(self._screen.width * 0.94)
                self.tap(send_x, self._screen.message_input_y)

        time.sleep(1)
        print("    Message sent")
        return True

    def attach_photo_to_dm(self, photo_path: str) -> bool:
        """Attach a photo to current DM conversation.

        Args:
            photo_path: Path to photo file on device or local.

        Returns:
            True if photo was attached.
        """
        print(f"  [DM] Attaching photo: {photo_path}")

        # Tap gallery button
        gallery_elem = self.find_element_by_text("Gallery", partial=True)
        if not gallery_elem:
            gallery_elem = self.find_element_by_id("gallery_button")

        if gallery_elem:
            self.tap_element(gallery_elem, delay=2)
        else:
            # Tap typical gallery icon location (left of input)
            gallery_x = int(self._screen.width * 0.08)
            self.tap(gallery_x, self._screen.message_input_y, delay=2)

        # Wait for gallery to open and select first/recent photo
        time.sleep(2)

        # Tap first photo in gallery grid
        # Typically starts around y=300 after header (use percentage)
        photo_x = int(self._screen.width * 0.14)
        photo_y = int(self._screen.height * 0.31)
        self.tap(photo_x, photo_y, delay=1)

        # Look for send/confirm button
        send_elem = self.find_element_by_text("Send", partial=True)
        if send_elem:
            self.tap_element(send_elem)
        else:
            # May auto-send on selection
            pass

        time.sleep(1)
        print("    Photo attached")
        return True

    def send_dm_with_photo(self, text: str, photo_path: str) -> bool:
        """Send a DM with both text and photo.

        Args:
            text: Message text.
            photo_path: Path to photo file.

        Returns:
            True if both were sent.
        """
        # Send text first
        text_sent = self.send_dm_message(text)

        # Then attach photo
        photo_sent = self.attach_photo_to_dm(photo_path)

        return text_sent and photo_sent

    def wait_for_new_message(self, timeout: int = 30, poll_interval: float = 2.0) -> Optional[Dict]:
        """Wait for a new incoming message in current conversation.

        Args:
            timeout: Max seconds to wait.
            poll_interval: Seconds between checks.

        Returns:
            New message dict or None if timeout.
        """
        print(f"  [DM] Waiting for new message (timeout={timeout}s)...")

        # Get current last message
        current_last = self.get_latest_message()
        current_content = current_last.get('content') if current_last else None

        elapsed = 0
        while elapsed < timeout:
            time.sleep(poll_interval)
            elapsed += poll_interval

            new_last = self.get_latest_message()
            if new_last:
                new_content = new_last.get('content')
                # Check if different and from user
                if new_content != current_content and new_last.get('role') == 'user':
                    print(f"    New message received: {new_content[:50]}...")
                    return new_last

        print("    Timeout waiting for message")
        return None

    def is_in_conversation(self) -> bool:
        """Check if currently in a DM conversation view.

        Returns:
            True if in conversation (vs inbox or other screen).
        """
        # Look for message input field
        input_elem = self.find_element_by_id("row_thread_composer_edittext")
        if input_elem:
            return True

        # Look for "Message..." placeholder
        input_elem = self.find_element_by_text("Message...", partial=True)
        return input_elem is not None

    def is_in_dm_inbox(self) -> bool:
        """Check if currently in DM inbox view.

        Returns:
            True if in inbox.
        """
        # Look for inbox indicators
        elements, _ = self.dump_ui()
        for elem in elements:
            text = elem.get('text', '').lower()
            desc = elem.get('desc', '').lower()
            if 'messages' in text or 'requests' in text:
                return True
            if 'direct' in desc and 'inbox' in desc:
                return True
        return False


if __name__ == "__main__":
    # Demo: would need actual Appium connection
    print("AppiumUIController - DM automation module")
    print("Requires connected Appium driver to function")
    print("\nDM Methods available:")
    print("  - open_dm_inbox()")
    print("  - get_dm_conversations()")
    print("  - get_unread_dms()")
    print("  - open_conversation(username)")
    print("  - read_conversation_messages()")
    print("  - get_latest_message()")
    print("  - send_dm_message(text)")
    print("  - attach_photo_to_dm(photo_path)")
    print("  - send_dm_with_photo(text, photo_path)")
    print("  - wait_for_new_message(timeout)")
